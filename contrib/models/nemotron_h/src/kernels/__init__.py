# SPDX-License-Identifier: Apache-2.0
"""NKI helpers for NemotronH.

``state_cache.py`` is copied unchanged from ``contrib/models/qwen3_5`` (PR #68):
gather/scatter of whole recurrent-state pages, which replaces the per-element
DMAs that ``index_select``/``index_copy_`` over state pages lower to on device.
"""
