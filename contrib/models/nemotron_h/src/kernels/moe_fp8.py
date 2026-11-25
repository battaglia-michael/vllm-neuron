"""FP8 routed-expert decode kernels (R2-05, ``NEMOTRON_H_FP8=1``): weights-stationary matmuls on FP8 tiles.

Storage (see ``moe.py``): float8_e4m3 (trn2 ``nl.float8_e4m3``, |q| <= 240), bf16 activations, fp32 PSUM.

* ``up_t`` ``[E * 128, nh * I]``: the up weight already transposed, partition-major per expert:
  ``up_t[e * 128 + p, hh * I + i] = Wq_up[e][i, hh * 128 + p]`` (H = nh * 128). One DMA moves an expert's whole
  ``upT [128 (h), nh, I]`` tile set into SBUF (9,744 contiguous bytes per partition), so no on-chip transposes.
* ``down`` ``[E * I, H]`` (in, out), natural row tiles ``[pt, H]``.
* ``c`` ``[E * I + 128, 1]`` float32, ``c_i = s_up_i^2 * s_down_i`` per intermediate row (``relu(s u)^2 = s^2 relu(u)^2``).

Per expert ``e`` (the up matmuls have the FP8 weight tile as the stationary operand and the few token
columns as the moving operand, which gives uT with I on partitions, the layout the down matmul needs):

    uT[i, t]  = sum_hh upT[:, hh, i]^T @ xT[:, hh, t]               # [I, T] fp32 PSUM, I on partitions
    aT[i, t]  = relu(uT)^2 * c_i * w[t, e]                          # bf16
    y[t, h]   += sum_j aT[j rows, t]^T @ down[j rows, h]            # [T, H] fp32 PSUM, accumulated over experts

``moe_fp8_topk``: the T * k selected (token, expert) pairs, runtime expert offsets (software DGE), pairs split over
the two physical cores. ``moe_fp8_dense``: every expert (static addresses), routing weight 0 for the unselected
ones (the dense decode path of the bf16 model, now reading half the bytes), experts split over the two cores.
Both return ``[n_prgs, T, H]`` float32 partials; the caller sums them. The down projection runs as
``y[T, H] += aT^T @ down`` (aT stationary, FP8 down tiles moving, one PSUM bank per 512 columns).
"""

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

PMAX = 128
_PSUM_COLS = 512
import os as _os

_DBG = set(filter(None, _os.environ.get("NEMOTRON_H_FP8_DBG", "").split(",")))  # debug knobs (unit tests)


def _geom(geom, h):
    rows, pt = geom.shape
    nj = (rows + pt - 1) // pt
    pts = [min(pt, rows - j * pt) for j in range(nj)]
    nh = h // PMAX
    return rows, pt, nj, pts, nh


def _load_xT(xt, nh, t_n):
    """``xt`` ``[H, T]`` bf16 (x transposed by the caller) -> SBUF ``[128, nh, T]``."""
    xT = nl.ndarray((PMAX, nh, t_n), dtype=xt.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xT, src=xt.ap(pattern=[[t_n, PMAX], [PMAX * t_n, nh], [1, t_n]], offset=0))
    return xT


def _expert(upT, dn_sb, c_sb, xT, scale_tt, yT, first, rows, pt, nj, pts, nh, t_n, h, scale_mode, scale_col=0):
    """One expert: uT -> aT -> accumulate yT (``yT`` PSUM ``[128, nh, T]``). ``scale_mode`` 'col': ``scale_tt``
    ``[pt, T]`` per-token factor tile; 'part': column ``scale_col`` of ``scale_tt`` (one routing weight broadcast
    to every partition)."""
    aT = nl.ndarray((pt, nj, t_n), dtype=xT.dtype, buffer=nl.sbuf)
    for j in range(nj):
        sz = pts[j]
        pu = nl.ndarray((pt, t_n), dtype=nl.float32, buffer=nl.psum)
        for hh in range(nh):
            nisa.nc_matmul(dst=pu[0:sz, 0:t_n], stationary=upT[0:PMAX, hh, j * pt : j * pt + sz],
                           moving=xT[0:PMAX, hh, 0:t_n], accumulate=hh > 0)
        r = nl.ndarray((pt, t_n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r[0:sz, 0:t_n], data=pu[0:sz, 0:t_n], op0=nl.maximum, operand0=0.0)
        a_f = nl.ndarray((pt, t_n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=a_f[0:sz, 0:t_n], data=r[0:sz, 0:t_n], op0=nl.multiply,
                                  operand0=c_sb[0:sz, j : j + 1], op1=nl.multiply, operand1=r[0:sz, 0:t_n])
        if scale_mode == "col":
            nisa.tensor_tensor(dst=aT[0:sz, j, 0:t_n], data1=a_f[0:sz, 0:t_n], data2=scale_tt[0:sz, 0:t_n],
                               op=nl.multiply)
        else:
            nisa.tensor_scalar(dst=aT[0:sz, j, 0:t_n], data=a_f[0:sz, 0:t_n], op0=nl.multiply,
                               operand0=scale_tt[0:sz, scale_col : scale_col + 1])
    # y[T, H] += aT^T @ down: one PSUM tile (bank) per 512 columns. Slices of one PSUM tensor that share a bank
    # cannot be used as separate accumulation targets: an overwrite (accumulate=False) of one slice cleared the
    # others (first expert of each core wrong, R2-05 probe), so each target is its own tile, as in moe_topk.py.
    for ci in range(len(yT)):
        c0 = ci * _PSUM_COLS
        csz = min(_PSUM_COLS, h - c0)
        for j in range(nj):
            sz = pts[j]
            nisa.nc_matmul(dst=yT[ci][0:t_n, 0:csz], stationary=aT[0:sz, j, 0:t_n],
                           moving=dn_sb[0:sz, j, c0 : c0 + csz], accumulate=not (first and j == 0))


def _psum_y(t_n, h):
    return [nl.ndarray((t_n, min(_PSUM_COLS, h - c0)), dtype=nl.float32, buffer=nl.psum)
            for c0 in range(0, h, _PSUM_COLS)]


def _store(out, prg_id, yT, nh, t_n, h, have):
    o_sb = nl.ndarray((t_n, h), dtype=nl.float32, buffer=nl.sbuf)
    if have:
        for ci in range(len(yT)):
            c0 = ci * _PSUM_COLS
            csz = min(_PSUM_COLS, h - c0)
            nisa.tensor_copy(dst=o_sb[0:t_n, c0 : c0 + csz], src=yT[ci][0:t_n, 0:csz])
    else:
        nisa.memset(dst=o_sb, value=0.0)
    nisa.dma_copy(dst=out[prg_id, 0:t_n, 0:h], src=o_sb)


@nki.jit
def moe_fp8_topk_kernel(xt: Tensor, up_t: Tensor, down: Tensor, erow: Tensor, eup: Tensor, w: Tensor,
                        geom: Tensor, c: Tensor):
    """Top-k pairs. ``xt`` ``[H, T]`` bf16; ``erow`` ``[n, 1]`` int32 = e * I (down / c rows); ``eup`` ``[n, 1]`` int32
    = e * 128 (up_t rows); ``w`` ``[1, n]`` float32, pair-major (pair p -> token p // k). Returns ``[n_prgs, H, T]``."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("moe_fp8_topk", (0, 1), 2)
    h, t_n = xt.shape
    n = erow.shape[0]
    k = n // t_n
    rows, pt, nj, pts, nh = _geom(geom, h)
    out = nl.ndarray((n_prgs, t_n, h), dtype=nl.float32, buffer=nl.shared_hbm)
    per = (n + n_prgs - 1) // n_prgs
    p_lo = prg_id * per
    p_hi = min(n, p_lo + per)

    er_sb = nl.ndarray((n, 1), dtype=erow.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=er_sb, src=erow)
    eu_sb = nl.ndarray((n, 1), dtype=eup.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=eu_sb, src=eup)
    w_b = nl.ndarray((pt, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_b, src=w.ap(pattern=[[0, pt], [1, n]], offset=0))
    xT = _load_xT(xt, nh, t_n)
    yT = _psum_y(t_n, h)
    sw = dict(dge_mode=nisa.dge_mode.swdge)
    for p in range(p_lo, p_hi):
        tok = p // k
        upT = nl.ndarray((PMAX, nh, rows), dtype=up_t.dtype, buffer=nl.sbuf)
        if "up_hw" in _DBG:
            nisa.dma_copy(dst=upT, src=up_t.ap(pattern=[[nh * rows, PMAX], [rows, nh], [1, rows]], offset=0,
                                               scalar_offset=eu_sb[p : p + 1, 0:1], indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        elif "up_split" in _DBG:
            for hh in range(nh):
                nisa.dma_copy(dst=upT[0:PMAX, hh, 0:rows],
                              src=up_t.ap(pattern=[[nh * rows, PMAX], [1, rows]], offset=hh * rows,
                                          scalar_offset=eu_sb[p : p + 1, 0:1], indirect_dim=0), **sw)
        else:
            nisa.dma_copy(dst=upT, src=up_t.ap(pattern=[[nh * rows, PMAX], [rows, nh], [1, rows]], offset=0,
                                               scalar_offset=eu_sb[p : p + 1, 0:1], indirect_dim=0), **sw)
        dn_sb = nl.ndarray((pt, nj, h), dtype=down.dtype, buffer=nl.sbuf)
        for j in range(nj):
            nisa.dma_copy(dst=dn_sb[0:pts[j], j, 0:h],
                          src=down.ap(pattern=[[h, pts[j]], [1, h]], offset=j * pt * h,
                                      scalar_offset=er_sb[p : p + 1, 0:1], indirect_dim=0), **sw)
        c_sb = nl.ndarray((pt, nj), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=c_sb, src=c.ap(pattern=[[1, pt], [pt, nj]], offset=0,
                                         scalar_offset=er_sb[p : p + 1, 0:1], indirect_dim=0), **sw)
        if t_n == 1 and "col" not in _DBG:
            _expert(upT, dn_sb, c_sb, xT, w_b, yT, p == p_lo, rows, pt, nj, pts, nh, 1, h, "part", scale_col=p)
        else:  # only this pair's token column is non-zero
            sc = nl.ndarray((pt, t_n), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=sc, value=0.0)
            nisa.tensor_copy(dst=sc[0:pt, tok : tok + 1], src=w_b[0:pt, p : p + 1])
            _expert(upT, dn_sb, c_sb, xT, sc, yT, p == p_lo, rows, pt, nj, pts, nh, t_n, h, "col")
    _store(out, prg_id, yT, nh, t_n, h, p_hi > p_lo)
    return out


@nki.jit
def moe_fp8_dense_kernel(xt: Tensor, up_t: Tensor, down: Tensor, affT: Tensor, geom: Tensor, c: Tensor):
    """All experts. ``affT`` ``[E, T]`` float32 routing weights (0 off the top-k). Experts ``prg_id, prg_id + 2, ...``
    on each physical core. Returns ``[n_prgs, H, T]``."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("moe_fp8_dense", (0, 1), 2)
    h, t_n = xt.shape
    e_n = affT.shape[0]
    rows, pt, nj, pts, nh = _geom(geom, h)
    out = nl.ndarray((n_prgs, t_n, h), dtype=nl.float32, buffer=nl.shared_hbm)
    xT = _load_xT(xt, nh, t_n)
    yT = _psum_y(t_n, h)
    mine = list(range(prg_id, e_n, n_prgs))
    for q, e in enumerate(mine):
        upT = nl.ndarray((PMAX, nh, rows), dtype=up_t.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=upT, src=up_t.ap(pattern=[[nh * rows, PMAX], [rows, nh], [1, rows]],
                                           offset=e * PMAX * nh * rows))
        dn_sb = nl.ndarray((pt, nj, h), dtype=down.dtype, buffer=nl.sbuf)
        for j in range(nj):
            nisa.dma_copy(dst=dn_sb[0:pts[j], j, 0:h], src=down[e * rows + j * pt : e * rows + j * pt + pts[j], 0:h])
        c_sb = nl.ndarray((pt, nj), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=c_sb, src=c.ap(pattern=[[1, pt], [pt, nj]], offset=e * rows))
        a_b = nl.ndarray((pt, t_n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_b, src=affT.ap(pattern=[[0, pt], [1, t_n]], offset=e * t_n))
        _expert(upT, dn_sb, c_sb, xT, a_b, yT, q == 0, rows, pt, nj, pts, nh, t_n, h, "col")
    _store(out, prg_id, yT, nh, t_n, h, len(mine) > 0)
    return out


def _tiles(rows: int, x: Tensor) -> Tensor:
    return torch.zeros((rows, min(rows, PMAX)), dtype=torch.int32, device=x.device)


def moe_fp8_topk(x: Tensor, up_t: Tensor, down: Tensor, c: Tensor, experts: Tensor, w: Tensor, rows: int) -> Tensor:
    """``x`` ``[T, H]``, ``experts`` / ``w`` ``[T, k]``. Returns the routed sum ``[T, H]`` in ``x.dtype``."""
    e32 = experts.to(torch.int32).reshape(-1, 1)
    out = wrap_nki(moe_fp8_topk_kernel)[2](
        xt=x.t().contiguous(), up_t=up_t, down=down, erow=e32 * rows, eup=e32 * PMAX,
        w=w.to(torch.float32).reshape(1, -1), geom=_tiles(rows, x), c=c)
    return out.sum(0).to(x.dtype)


def moe_fp8_dense(x: Tensor, up_t: Tensor, down: Tensor, c: Tensor, affinity: Tensor, rows: int) -> Tensor:
    """``affinity`` ``[T, E]`` float32 (0 off the top-k). Returns ``[T, H]`` in ``x.dtype``."""
    out = wrap_nki(moe_fp8_dense_kernel)[2](
        xt=x.t().contiguous(), up_t=up_t, down=down, affT=affinity.to(torch.float32).t().contiguous(),
        geom=_tiles(rows, x), c=c)
    return out.sum(0).to(x.dtype)


def up_t_layout(q_up: Tensor, num_experts: int, rows: int) -> Tensor:
    """``[E * I, H]`` (out, in) -> ``[E * 128, nh * I]`` (see module docstring). Works on any dtype (CPU)."""
    h = q_up.shape[1]
    nh = h // PMAX
    v = q_up.view(torch.uint8) if q_up.dtype == torch.float8_e4m3fn else q_up
    v = v.view(num_experts, rows, nh, PMAX).permute(0, 3, 2, 1).contiguous().view(num_experts * PMAX, nh * rows)
    return v.view(q_up.dtype) if q_up.dtype == torch.float8_e4m3fn else v


def up_from_t_layout(up_t: Tensor, num_experts: int, rows: int) -> Tensor:
    """Inverse of ``up_t_layout`` (torch, for the CPU reference path): ``[E * I, H]``."""
    nh = up_t.shape[1] // rows
    return up_t.view(num_experts, PMAX, nh, rows).permute(0, 3, 2, 1).reshape(num_experts * rows, nh * PMAX)
