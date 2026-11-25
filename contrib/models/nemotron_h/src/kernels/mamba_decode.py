"""Fused Mamba-2 decode step for NemotronH (one layer, after ``in_proj``).

Replaces, for one decode step of a batch of requests, everything between the
``in_proj`` matmul and the ``out_proj`` matmul in
``NemotronHMamba2Mixer.forward_decode``:

    state page read (conv window + SSM state)
    xBC   = silu(depthwise_conv4(window, xBC) + conv_bias)      (x, B, C rounded to bf16 as torch does)
    dt    = softplus(dt + dt_bias)                              (no clamp, NEMOTRON_H_DT_CLAMP=none)
    dA    = exp(dt * A)
    h     = h * dA + (dt * x) B                                 (fp32, written back in place)
    y     = C . h + D x
    y     = RMSNormGated(y, z)  (y * silu(z), RMS over 512-channel groups, * weight) -> bf16
    state page write (shifted conv window + new SSM state)

Layout. Each SSM group (8 heads x 64 channels = 512 state rows of d_state=128
floats, plus its 512 x / 128 B / 128 C conv channels and its 512-channel norm
group) is independent, so with an LNC=2 launch program ``g`` does group ``g``.
Inside a group:

* row phase, requests on partitions: conv, dt, dA, dt*x, gating and norm;
* state phase, one request at a time: partition ``q`` holds state rows
  ``4q .. 4q+3`` of the group (head ``q // 16``), d_state on the free axis, so
  a request's 256 KiB group state is one 2 KiB contiguous run per partition,
  ``dA`` is one per-partition scalar and ``dt*x`` is one per (partition, row);
  ``B`` and ``C`` are broadcast to all partitions with one bf16 matmul against
  a one-hot selector (exact), the readout ``y = C . h`` is a fused
  multiply-reduce on the Vector engine;
* TensorE transposes (bit exact on trn2) move ``dA`` / ``dt*x`` into the state
  layout and ``y`` back.

Paged state: the page of request ``i`` is ``pages[i]`` in ``state_a`` /
``state_b`` (``[num_pages, rows, 128]`` float32 halves, page = conv window
``[3, conv_dim]`` then SSM state ``[heads, head_dim, d_state]``, spilling from
half A into half B; with ``NEMOTRON_H_STATE_ALIGN=1`` half A holds the conv window
and group 0, half B group 1 from row 0, see ``ssm_rows_a``). Writes use ``wpages``, where padding rows carry an
out-of-range id so their write is skipped (``oob_mode.skip``); the torch path
writes zeros to the null page 0 instead, which nothing reads.
"""

import os

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from .state_cache import f32_pages

R = 128  # floats per state row (the [pages, rows, 128] view)
ROWS_PER_PART = 4  # state rows per partition in the state phase


_ENGINES = (nisa.engine.sync, nisa.engine.scalar)


def _dma(dst, src, to_hbm, k, eng_mode=0):
    """Dynamic-offset DMA on hardware DGE.

    ``eng_mode`` picks the issuing sequencer: 0 alternates Sync / Scalar by ``k``,
    1 uses Sync only, 2 reads on Sync and writes on Scalar, 3 software DGE (GpSimd),
    4 alternates Sync hardware DGE and software DGE by ``k``, 5 reads on software
    DGE and writes on Sync hardware DGE, 6 reads on Sync and writes on software DGE.
    (:func:`_state_dma` also takes 7 / 8: state half A on software / hardware DGE
    and half B on the other one.)
    """
    if eng_mode == 5:
        eng_mode = 1 if to_hbm else 3
    elif eng_mode == 6:
        eng_mode = 3 if to_hbm else 1
    if eng_mode == 3 or (eng_mode == 4 and k % 2):
        if to_hbm:
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.swdge, oob_mode=oob_mode.skip)
        else:
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.swdge)
        return
    if eng_mode in (1, 4):
        eng = nisa.engine.sync
    elif eng_mode == 2:
        eng = nisa.engine.scalar if to_hbm else nisa.engine.sync
    else:
        eng = _ENGINES[k % 2]
    if to_hbm:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, oob_mode=oob_mode.skip, engine=eng)
    else:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=eng)


def _seg_rows_dma(st, state, idx, page_row0, rel0, rel1, to_hbm, split=1, k0=0, eng_mode=0):
    """Move group-relative state rows ``[rel0, rel1)`` between SBUF ``st`` and one page.

    ``st`` is ``[128, 4 * R]`` (partition ``q`` = relative rows ``4q .. 4q+3``);
    relative row ``r`` is row ``page_row0 + r`` of ``state``'s page ``idx``.
    Emits one DMA for the whole-partition middle and one per partial partition.
    """
    _, rows, _ = state.shape
    pieces = []
    a, b = rel0, rel1
    head_end = min(b, (a + ROWS_PER_PART - 1) // ROWS_PER_PART * ROWS_PER_PART)
    if a % ROWS_PER_PART and a < head_end:
        pieces.append((a, head_end))
        a = head_end
    mid_end = b // ROWS_PER_PART * ROWS_PER_PART
    if a < mid_end:
        # Whole partitions, cut into ``split`` DMAs (spread over the DGE queues).
        step = -(-(mid_end - a) // (split * ROWS_PER_PART)) * ROWS_PER_PART
        while a < mid_end:
            pieces.append((a, min(mid_end, a + step)))
            a = min(mid_end, a + step)
    if a < b:
        pieces.append((a, b))
    for k, (p0, p1) in enumerate(pieces):
        part0 = p0 // ROWS_PER_PART
        if p0 % ROWS_PER_PART == 0 and p1 % ROWS_PER_PART == 0:
            nparts = (p1 - p0) // ROWS_PER_PART
            sb = st[part0 : part0 + nparts, 0 : ROWS_PER_PART * R]
            hbm = state.ap(
                pattern=[[ROWS_PER_PART * R, nparts], [1, ROWS_PER_PART * R]],
                offset=(page_row0 + p0) * R,
                scalar_offset=idx,
                indirect_dim=0,
            )
        else:
            j0 = p0 % ROWS_PER_PART
            j1 = j0 + (p1 - p0)
            sb = st[part0 : part0 + 1, j0 * R : j1 * R]
            hbm = state.ap(
                pattern=[[ROWS_PER_PART * R, 1], [1, (p1 - p0) * R]],
                offset=(page_row0 + p0) * R,
                scalar_offset=idx,
                indirect_dim=0,
            )
        if to_hbm:
            _dma(hbm, sb, True, k0 + k, eng_mode)
        else:
            _dma(sb, hbm, False, k0 + k, eng_mode)
    return len(pieces)


def _group_start(half_rows, conv_rows, g, group_rows, ssm_rows_a=-1):
    """Page row of group ``g``'s first SSM row (page rows ``>= half_rows`` are in half B).

    ``ssm_rows_a`` = SSM rows stored in half A after the conv window. -1 (default
    layout): A is filled, so the SSM state is contiguous from page row ``conv_rows``
    and a group may straddle the halves. ``>= 0`` (``NEMOTRON_H_STATE_ALIGN=1``):
    SSM rows ``[0, ssm_rows_a)`` are in A from row ``conv_rows``, the rest in B from
    row 0, and ``ssm_rows_a`` is a multiple of ``group_rows``, so every group lies in
    one half.
    """
    r0 = g * group_rows
    if ssm_rows_a < 0 or r0 < ssm_rows_a:
        return conv_rows + r0
    return half_rows + r0 - ssm_rows_a


def _vec_pieces(state_a, state_b, conv_rows, g, group_rows, ssm_rows_a=-1):
    """Group ``g``'s SSM rows as ``(state half, first partition, end partition, row shift)``.

    Partition ``q`` of the state tile holds group-relative rows ``4q .. 4q+3``,
    i.e. rows ``4q + shift`` of the half (``shift`` may be negative for half B).
    """
    _, half_rows, _ = state_a.shape
    start = _group_start(half_rows, conv_rows, g, group_rows, ssm_rows_a)
    end = start + group_rows
    a_end = min(end, half_rows)
    pieces = []
    if start < a_end:
        pieces.append((state_a, 0, (a_end - start) // ROWS_PER_PART, start))
    b_start = max(start, half_rows)
    if b_start < end:
        pieces.append((state_b, (b_start - start) // ROWS_PER_PART, group_rows // ROWS_PER_PART, start - half_rows))
    return pieces


def _vec_offsets(idx_hbm, n, rows_pp, q0, q1, shift, q4f):
    """``[q1 - q0, n]`` int32: ``idx[i] * rows_pp + 4 (q0 + p) + shift`` on partition ``p``, column ``i``.

    Starts at partition 0: a vector-DGE index tensor must (NCC_EBIR026).
    """
    m = q1 - q0
    pb = nl.ndarray((m, n), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pb, src=idx_hbm.ap(pattern=[[0, m], [1, n]], offset=0))
    pf = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pf, src=pb)
    of = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=of, data=pf, op0=nl.multiply, operand0=float(rows_pp), op1=nl.add, operand1=q4f[0:m, 0:1])
    if shift + ROWS_PER_PART * q0:
        nisa.tensor_scalar(dst=of, data=of, op0=nl.add, operand0=float(shift + ROWS_PER_PART * q0))
    oi = nl.ndarray((m, n), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=oi, src=of)
    return oi


def _state_dma_vec(st, vec, i, to_hbm):
    """Request ``i``'s group state as per-partition (vector-offset) software-DGE DMAs.

    ``vec`` is a list of ``(state half, q0, q1, offsets [128, n])`` (see
    :func:`_vec_pieces`); the halves are addressed as ``[num_pages * rows, 128]``
    row arrays, so partition ``q`` moves its 4 contiguous rows (2 KiB) on its own.
    An out-of-range write page gives out-of-range rows, whose transfer is skipped.
    """
    for state, q0, q1, offs in vec:
        flat = state.reshape((state.shape[0] * state.shape[1], R))
        hbm = flat.ap(
            pattern=[[R, q1 - q0], [1, ROWS_PER_PART * R]],
            offset=0,
            vector_offset=offs[0 : q1 - q0, i : i + 1],
            indirect_dim=0,
        )
        sb = st[q0:q1, 0 : ROWS_PER_PART * R]
        if to_hbm:
            nisa.dma_copy(dst=hbm, src=sb, dge_mode=nisa.dge_mode.swdge, oob_mode=oob_mode.skip)
        else:
            nisa.dma_copy(dst=sb, src=hbm, dge_mode=nisa.dge_mode.swdge)


def _state_vec_ok(state_a, conv_rows, g, group_rows, ssm_rows_a=-1):
    """Mode 9 needs every state-half boundary of the group on a partition boundary."""
    half_rows = state_a.shape[1]
    start = _group_start(half_rows, conv_rows, g, group_rows, ssm_rows_a)
    return start % ROWS_PER_PART == 0 and (half_rows - start) % ROWS_PER_PART == 0


def _state_dma(st, state_a, state_b, idx, conv_rows, g, group_rows, to_hbm, split=1, k0=0, eng_mode=0, sw_parts=64,
               ssm_rows_a=-1):
    """Group ``g``'s SSM rows of one page, split over the two halves."""
    _, half_rows, _ = state_a.shape
    start = _group_start(half_rows, conv_rows, g, group_rows, ssm_rows_a)  # page row of relative row 0
    end = start + group_rows
    # Rows in half A: page rows [start, min(end, half_rows)).
    a_end = min(end, half_rows)
    if eng_mode == 10:
        # Partitions [0, sw_parts) of the group on software DGE, the rest on Sync
        # hardware DGE (two DGE paths per core), cut also at the half A / B boundary.
        cut = sw_parts * ROWS_PER_PART
        k = k0
        for st_half, row0, lo, hi in ((state_a, start, 0, a_end - start), (state_b, start - half_rows, max(start, half_rows) - start, group_rows)):
            for r0, r1, mode in ((lo, min(hi, cut), 3), (max(lo, cut), hi, 1)):
                if r0 < r1:
                    k += _seg_rows_dma(st, st_half, idx, row0, r0, r1, to_hbm, 1, k, mode)
        return
    k = k0
    mode_a = mode_b = eng_mode
    if eng_mode in (7, 8):  # half A / half B pieces on different DGE modes
        mode_a, mode_b = (3, 1) if eng_mode == 7 else (1, 3)
    if start < a_end:
        k += _seg_rows_dma(st, state_a, idx, start, 0, a_end - start, to_hbm, split, k, mode_a)
    # Rows in half B: page rows [max(start, half_rows), end), B row = page row - half_rows.
    b_start = max(start, half_rows)
    if b_start < end:
        _seg_rows_dma(st, state_b, idx, start - half_rows, b_start - start, group_rows, to_hbm, split, k, mode_b)


@nki.jit
def mamba2_decode_kernel(
    proj: Tensor,
    state_a: Tensor,
    state_b: Tensor,
    pages: Tensor,
    wpages: Tensor,
    conv_w: Tensor,
    conv_b: Tensor,
    dt_bias: Tensor,
    a_neg: Tensor,
    d_skip: Tensor,
    norm_w: Tensor,
    n_groups: int,
    head_dim: int,
    d_state: int,
    eps: float,
    ablate: int = 0,
    dma_split: int = 1,
    win_vec: int = 1,
    dma_eng: int = 1,
    prefetch: int = 2,
    state_rows: int = 0,
    dma_eng1: int = -1,
    dma_sw_parts: int = 64,
    ssm_rows_a: int = -1,
):
    """One Mamba-2 decode step for ``n`` requests (see module docstring).

    Args:
        proj: ``[n, I + conv_dim + H]`` bf16, ``in_proj`` output ``[z | xBC | dt]``.
        state_a, state_b: ``[num_pages, rows, 128]`` float32 state halves (updated in place).
        pages: ``[n, 1]`` int32 page to read per request.
        wpages: ``[n, 1]`` int32 page to write (out of range = skip).
        conv_w: ``[conv_dim, 4]`` bf16. conv_b: ``[conv_dim]`` bf16.
        dt_bias, a_neg, d_skip: ``[H]`` float32 (``a_neg = -exp(A_log)``).
        norm_w: ``[I]`` bf16.
        ssm_rows_a: SSM state rows in half A (see :func:`_group_start`); -1 = default
            layout (A filled after the conv window, group 1 straddles the halves).
        state_rows: 0, or the rows per half when ``state_a`` / ``state_b`` are the 2-byte
            K / V tensors of the shared pool (``state_cache.f32_pages`` bit-casts them).

    Returns:
        ``(y [n, I] bf16, state_a, state_b)``; ``y`` is the gated-norm output, the
        ``out_proj`` input.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("mamba2_decode", (0, 1), 2)
    out_a, out_b = state_a, state_b
    state_a, state_b = f32_pages(state_a, state_rows), f32_pages(state_b, state_rows)
    n, proj_size = proj.shape
    nh = dt_bias.shape[0]
    P, N, G = head_dim, d_state, n_groups
    I = nh * P
    gi = I // G  # channels (= norm group) per SSM group
    hpg = nh // G
    conv_dim = conv_w.shape[0]
    K = conv_w.shape[1]
    KW = K - 1
    gbc = G * N
    group_rows = gi  # state rows per group: hpg * P rows of N floats (N == R)
    conv_rows = KW * conv_dim // R
    assert N == R and conv_dim * KW % R == 0 and gi % 128 == 0
    assert proj_size == I + conv_dim + nh and n <= 128
    assert group_rows == 128 * ROWS_PER_PART
    assert conv_rows < state_a.shape[1]
    if ssm_rows_a >= 0:
        assert ssm_rows_a % group_rows == 0 and conv_rows + ssm_rows_a <= state_a.shape[1]
        assert G * group_rows - ssm_rows_a <= state_b.shape[1]
    parts_per_head = 128 // hpg  # partitions per head in the state phase

    y_out = nl.ndarray((n, I), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # ── Shared inputs ────────────────────────────────────────────────
    proj_sb = nl.ndarray((n, proj_size), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=proj_sb, src=proj[0:n, 0:proj_size])
    idx_sb = nl.ndarray((n, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_sb, src=pages[0:n, 0:1])
    widx_sb = nl.ndarray((n, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=widx_sb, src=wpages[0:n, 0:1])
    # Parameters broadcast to every request partition (partition stride 0).
    w_sb = nl.ndarray((n, conv_dim * K), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_sb, src=conv_w.ap(pattern=[[0, n], [1, conv_dim * K]], offset=0))
    cb_sb = nl.ndarray((n, conv_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=cb_sb, src=conv_b.ap(pattern=[[0, n], [1, conv_dim]], offset=0))
    par_sb = nl.ndarray((n, 3, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=par_sb[0:n, 0, 0:nh], src=dt_bias.ap(pattern=[[0, n], [1, nh]], offset=0))
    nisa.dma_copy(dst=par_sb[0:n, 1, 0:nh], src=a_neg.ap(pattern=[[0, n], [1, nh]], offset=0))
    nisa.dma_copy(dst=par_sb[0:n, 2, 0:nh], src=d_skip.ap(pattern=[[0, n], [1, nh]], offset=0))
    nw_sb = nl.ndarray((n, I), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=nw_sb, src=norm_w.ap(pattern=[[0, n], [1, I]], offset=0))

    ident = nl.shared_identity_matrix(128, dtype=nl.float32)
    ones_pp = nl.ndarray((n, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones_pp, value=1.0)
    ones_f = nl.ndarray((n, parts_per_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)

    if 9 in (dma_eng, dma_eng1):
        q4 = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=q4, pattern=[[0, 1]], offset=0, channel_multiplier=ROWS_PER_PART)
        q4f = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q4f, src=q4)

    for g in range(G):
        if n_prgs > 1 and g % n_prgs != prg_id:
            continue
        # Conv channel segments of this group: (channel offset in xBC, width, offset in group tile).
        # State DMA mode of this group (``dma_eng1`` >= 0 overrides it for odd groups,
        # i.e. the second physical core, whose group straddles the state halves).
        g_eng = dma_eng1 if (g % 2 and dma_eng1 >= 0) else dma_eng
        if g_eng == 9 and not _state_vec_ok(state_a, conv_rows, g, group_rows, ssm_rows_a):
            g_eng = 1
        if g_eng == 9:
            # Mode 9: per-partition row offsets of every piece, for the read and the write pages.
            rvec, wvec = [], []
            for state, q0, q1, shift in _vec_pieces(state_a, state_b, conv_rows, g, group_rows, ssm_rows_a):
                rvec.append((state, q0, q1, _vec_offsets(pages, n, state.shape[1], q0, q1, shift, q4f)))
                wvec.append((state, q0, q1, _vec_offsets(wpages, n, state.shape[1], q0, q1, shift, q4f)))
        segs = ((g * gi, gi, 0), (I + g * N, N, gi), (I + gbc + g * N, N, gi + N))
        gw = gi + 2 * N

        # ── Row phase: conv window, conv, SiLU ─────────────────────
        win = nl.ndarray((n, KW, gw), dtype=nl.float32, buffer=nl.sbuf)
        if ablate & 1:
            nisa.memset(dst=win, value=0.0)
        if win_vec and not ablate & 1:
            # One gather per segment for all requests: per-partition page offsets (software DGE).
            for c0, w, so in segs:
                nisa.dma_copy(
                    dst=win[0:n, 0:KW, so : so + w],
                    src=state_a.ap(
                        pattern=[[state_a.shape[1] * R, n], [conv_dim, KW], [1, w]],
                        offset=c0, vector_offset=idx_sb, indirect_dim=0,
                    ),
                    dge_mode=nisa.dge_mode.swdge,
                )
        for i in range(n if not (ablate & 1 or win_vec) else 0):
            for si, (c0, w, so) in enumerate(segs):
                _dma(
                    win[i : i + 1, 0:KW, so : so + w],
                    state_a.ap(
                        pattern=[[state_a.shape[1] * R, 1], [conv_dim, KW], [1, w]],
                        offset=c0,
                        scalar_offset=idx_sb[i : i + 1, 0:1],
                        indirect_dim=0,
                    ),
                    False, i * 3 + si,
                )
        xc = nl.ndarray((n, gi), dtype=nl.bfloat16, buffer=nl.sbuf)  # silu(conv) of x
        bc = nl.ndarray((n, 2 * N), dtype=nl.bfloat16, buffer=nl.sbuf)  # [B | C]
        nwin = nl.ndarray((n, KW, gw), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=nwin[0:n, 0 : KW - 1, 0:gw], src=win[0:n, 1:KW, 0:gw])
        for c0, w, so in segs:
            xr = proj_sb[0:n, I + c0 : I + c0 + w]
            nisa.tensor_copy(dst=nwin[0:n, KW - 1, so : so + w], src=xr)
            acc = nl.ndarray((n, w), dtype=nl.float32, buffer=nl.sbuf)
            tmp = nl.ndarray((n, w), dtype=nl.float32, buffer=nl.sbuf)
            for k in range(K):
                wk = w_sb.ap(pattern=[[conv_dim * K, n], [K, w]], offset=c0 * K + k)
                src = win[0:n, k, so : so + w] if k < KW else xr
                if k == 0:
                    nisa.tensor_tensor(dst=acc, data1=src, data2=wk, op=nl.multiply)
                else:
                    nisa.tensor_tensor(dst=tmp, data1=src, data2=wk, op=nl.multiply)
                    nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=cb_sb[0:n, c0 : c0 + w], op=nl.add)
            if so == 0:
                nisa.activation(dst=xc, op=nl.silu, data=acc)
            else:
                nisa.activation(dst=bc[0:n, so - gi : so - gi + w], op=nl.silu, data=acc)
        # New conv window = old rows 1.. + this token's raw xBC.
        if win_vec and not ablate & 1:
            for c0, w, so in segs:
                nisa.dma_copy(
                    dst=state_a.ap(
                        pattern=[[state_a.shape[1] * R, n], [conv_dim, KW], [1, w]],
                        offset=c0, vector_offset=widx_sb, indirect_dim=0,
                    ),
                    src=nwin[0:n, 0:KW, so : so + w],
                    dge_mode=nisa.dge_mode.swdge,
                    oob_mode=oob_mode.skip,
                )
        for i in range(n if not (ablate & 1 or win_vec) else 0):
            for si, (c0, w, so) in enumerate(segs):
                _dma(
                    state_a.ap(
                        pattern=[[state_a.shape[1] * R, 1], [conv_dim, KW], [1, w]],
                        offset=c0,
                        scalar_offset=widx_sb[i : i + 1, 0:1],
                        indirect_dim=0,
                    ),
                    nwin[i : i + 1, 0:KW, so : so + w],
                    True, i * 3 + si,
                )

        # ── dt, dA, dt * x ────────────────────────────────────────
        h0 = g * hpg
        dtt = nl.ndarray((n, hpg), dtype=nl.float32, buffer=nl.sbuf)
        dt_col = I + conv_dim + h0
        nisa.tensor_tensor(dst=dtt, data1=proj_sb[0:n, dt_col : dt_col + hpg], data2=par_sb[0:n, 0, h0 : h0 + hpg], op=nl.add)
        dts = nl.ndarray((n, hpg), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=dts, op=nl.softplus, data=dtt)
        dtA = nl.ndarray((n, hpg), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dtA, data1=dts, data2=par_sb[0:n, 1, h0 : h0 + hpg], op=nl.multiply)
        dA = nl.ndarray((n, hpg), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=dA, op=nl.exp, data=dtA)
        dA16 = nl.ndarray((n, 128), dtype=nl.float32, buffer=nl.sbuf)
        dtx = nl.ndarray((n, gi), dtype=nl.float32, buffer=nl.sbuf)
        for hl in range(hpg):
            nisa.tensor_scalar(
                dst=dA16[0:n, hl * parts_per_head : (hl + 1) * parts_per_head],
                data=ones_f, op0=nl.multiply, operand0=dA[0:n, hl : hl + 1],
            )
            nisa.tensor_scalar(
                dst=dtx[0:n, hl * P : (hl + 1) * P],
                data=xc[0:n, hl * P : (hl + 1) * P], op0=nl.multiply, operand0=dts[0:n, hl : hl + 1],
            )
        # Into the state layout: dAq[q, i] = dA[i, head(q)], dtxE[q, j, i] = dtx[i, 4q + j].
        dAq = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
        dAq_ps = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=dAq_ps, data=dA16, engine=nisa.tensor_engine)
        nisa.tensor_copy(dst=dAq, src=dAq_ps)
        dtxE = nl.ndarray((128, ROWS_PER_PART, n), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(ROWS_PER_PART):
            t_ps = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(
                dst=t_ps,
                data=dtx.ap(pattern=[[gi, n], [ROWS_PER_PART, 128]], offset=j),
                engine=nisa.tensor_engine,
            )
            nisa.tensor_copy(dst=dtxE[0:128, j, 0:n], src=t_ps)

        # ── State phase, one request at a time ─────────────────────
        yE = nl.ndarray((128, ROWS_PER_PART, n), dtype=nl.float32, buffer=nl.sbuf)
        # Software pipeline: the read of request i + prefetch is issued before the
        # compute and write of request i, so a queue never holds a read behind a
        # write that waits for compute.
        sts = {}

        def _read(i):
            st = nl.ndarray((128, ROWS_PER_PART * R), dtype=nl.float32, buffer=nl.sbuf)
            if ablate & 2:
                nisa.memset(dst=st, value=0.0)
            else:
                if g_eng == 9:
                    _state_dma_vec(st, rvec, i, False)
                else:
                    _state_dma(st, state_a, state_b, idx_sb[i : i + 1, 0:1], conv_rows, g, group_rows, False, dma_split, i, g_eng, dma_sw_parts, ssm_rows_a)
            sts[i] = st

        for i in range(min(prefetch, n)):
            _read(i)
        for i in range(n):
            if i + prefetch < n:
                _read(i + prefetch)
            st = sts.pop(i)
            # Selector: row i all ones -> matmul broadcasts request i's [B | C] to 128 partitions.
            sel = nl.ndarray((n, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sel, data=ones_pp, op0=nl.multiply, operand0=ident[0:n, i : i + 1])
            bc_ps = nl.ndarray((128, 2 * N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=bc_ps, stationary=sel, moving=bc)
            bdtx = nl.ndarray((128, ROWS_PER_PART * R), dtype=nl.float32, buffer=nl.sbuf)
            for j in range(ROWS_PER_PART):
                nisa.activation(
                    dst=bdtx[0:128, j * R : (j + 1) * R], op=nl.copy, data=bc_ps[0:128, 0:N],
                    scale=dtxE[0:128, j, i : i + 1],
                )
            hn = nl.ndarray((128, ROWS_PER_PART * R), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=hn, data=st, op0=nl.multiply, operand0=dAq[0:128, i : i + 1], op1=nl.add, operand1=bdtx,
            )
            for j in range(ROWS_PER_PART):
                prod = nl.ndarray((128, R), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(
                    dst=prod, data=hn[0:128, j * R : (j + 1) * R], op0=nl.multiply, operand0=1.0,
                    op1=nl.multiply, operand1=bc_ps[0:128, N : 2 * N],
                    reduce_op=nl.add, reduce_res=yE[0:128, j, i : i + 1],
                )
            if not ablate & 2:
                if g_eng == 9:
                    _state_dma_vec(hn, wvec, i, True)
                else:
                    _state_dma(hn, state_a, state_b, widx_sb[i : i + 1, 0:1], conv_rows, g, group_rows, True, dma_split, i + 1, g_eng, dma_sw_parts, ssm_rows_a)

        # ── Back to rows: y + D x, gate, RMS norm ──────────────────
        y_rows = nl.ndarray((n, gi), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(ROWS_PER_PART):
            y_ps = nl.ndarray((n, 128), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=y_ps, data=yE[0:128, j, 0:n], engine=nisa.tensor_engine)
            nisa.tensor_copy(dst=y_rows.ap(pattern=[[gi, n], [ROWS_PER_PART, 128]], offset=j), src=y_ps)
        y2 = nl.ndarray((n, gi), dtype=nl.float32, buffer=nl.sbuf)
        for hl in range(hpg):
            nisa.scalar_tensor_tensor(
                dst=y2[0:n, hl * P : (hl + 1) * P], data=xc[0:n, hl * P : (hl + 1) * P],
                op0=nl.multiply, operand0=par_sb[0:n, 2, h0 + hl : h0 + hl + 1],
                op1=nl.add, operand1=y_rows[0:n, hl * P : (hl + 1) * P],
            )
        zs = nl.ndarray((n, gi), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=zs, op=nl.silu, data=proj_sb[0:n, g * gi : (g + 1) * gi])
        gt = nl.ndarray((n, gi), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gt, data1=y2, data2=zs, op=nl.multiply)
        sq = nl.ndarray((n, gi), dtype=nl.float32, buffer=nl.sbuf)
        ss = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=sq, op=nl.square, data=gt, reduce_op=nl.add, reduce_res=ss)
        var = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=var, data=ss, op0=nl.multiply, operand0=1.0 / gi, op1=nl.add, operand1=eps)
        rstd = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=var)
        out_bf = nl.ndarray((n, gi), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=out_bf, data=gt, op0=nl.multiply, operand0=rstd, op1=nl.multiply,
            operand1=nw_sb[0:n, g * gi : (g + 1) * gi],
        )
        nisa.dma_copy(dst=y_out[0:n, g * gi : (g + 1) * gi], src=out_bf)

    return y_out, out_a, out_b


# State-DMA routing (A05b). ``v2``: every state DMA on the Sync queue (hardware DGE).
# ``v3``: the same, except that the second core's group, which straddles the two state
# halves, moves its half-A piece on software DGE so its two DMAs per request run on
# two queues (kernel-only time -18% at batch 32, -20% at batch 128 on trn2).
DECODE_DMA = os.environ.get("NEMOTRON_H_MAMBA_DECODE_DMA", "v3")
_DMA_KWARGS = {"v2": {}, "v3": {"dma_eng": 1, "dma_eng1": 7}}
if DECODE_DMA not in _DMA_KWARGS:
    raise ValueError(f"NEMOTRON_H_MAMBA_DECODE_DMA must be one of {sorted(_DMA_KWARGS)}, got {DECODE_DMA!r}")


def kernel_args(mixer, proj: Tensor, pages: Tensor, wpages: Tensor) -> dict:
    """Keyword arguments of :func:`mamba2_decode_kernel` for a ``NemotronHMamba2Mixer``."""
    # Aligned page layout (NEMOTRON_H_STATE_ALIGN=1): passed only when on, so the
    # default kernel call is unchanged.
    rows_a = getattr(mixer, "ssm_rows_a", None)
    align = {} if rows_a is None else {"ssm_rows_a": rows_a}
    return dict(
        **_DMA_KWARGS[DECODE_DMA],
        **align,
        # R2-03p2 diagnostic: NEMOTRON_H_MAMBA_DECODE_WINVEC=0 moves the conv window per request on hardware DGE
        # (scalar offsets) instead of one software-DGE vector-offset gather/scatter; unset = the kernel default.
        **({"win_vec": int(os.environ["NEMOTRON_H_MAMBA_DECODE_WINVEC"])} if "NEMOTRON_H_MAMBA_DECODE_WINVEC" in os.environ else {}),
        proj=proj.contiguous(),
        state_a=mixer.state_a,
        state_b=mixer.state_b,
        pages=pages.to(torch.int32).view(-1, 1),
        wpages=wpages.to(torch.int32).view(-1, 1),
        conv_w=mixer.conv1d_weight,
        conv_b=mixer.conv1d_bias,
        dt_bias=mixer.dt_bias,
        a_neg=-torch.exp(mixer.A_log.float()),
        d_skip=mixer.D.float(),
        norm_w=mixer.norm_weight,
        n_groups=mixer.n_groups,
        head_dim=mixer.head_dim,
        d_state=mixer.d_state,
        eps=float(mixer.eps),
        **({"state_rows": mixer.state_shape[1]} if mixer.state_kv else {}),
    )


def fused_decode(mixer, proj: Tensor, pages: Tensor, wpages: Tensor) -> Tensor:
    """Run the kernel on device (LNC=2 grid, one SSM group per core); returns ``y [n, I]`` bf16."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    y, _, _ = wrap_nki(mamba2_decode_kernel)[2](**kernel_args(mixer, proj, pages, wpages))
    return y
