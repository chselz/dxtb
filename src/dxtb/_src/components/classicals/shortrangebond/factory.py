# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""Factory for the GFN0 short-range bond correction."""

from __future__ import annotations

import torch

from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import DD, Tensor, get_default_dtype

from ..gfn0 import legacy_d3_radii
from .srb import ShortRangeBond

__all__ = ["new_srb"]


def new_srb(
    unique: Tensor,
    par: Param | ParamModule,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ShortRangeBond | None:
    """Create the GFN0 SRB contribution when its parameter block is present."""
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    if not isinstance(par, ParamModule):
        par = ParamModule(par, **dd)

    if "short_range" not in par or par.is_none("short_range.srb"):
        return None
    if par.is_none("charge.eeq"):
        raise ValueError("The SRB correction requires GFN0 charge.eeq CN data.")
    if par.get("charge.eeq.cn") != "erf":
        raise ValueError("The GFN0 SRB correction only supports erf CN.")

    return ShortRangeBond(
        r0=par.get_elem_param(unique, "srb_r0", pad_val=0),
        cnfak=par.get_elem_param(unique, "srb_cnfak", pad_val=0),
        en=par.get_elem_param(unique, "srb_en", pad_val=0),
        rcov=legacy_d3_radii(**dd),
        shift=par.get("short_range.srb.shift"),
        prefactor=par.get("short_range.srb.prefactor"),
        steepness=par.get("short_range.srb.steepness"),
        enscale=par.get("short_range.srb.enscale"),
        period1=par.get("short_range.srb.period1"),
        period2=par.get("short_range.srb.period2"),
        cutoff2=par.get("short_range.srb.cutoff2"),
        cn_cutoff=par.get("charge.eeq.cutoff"),
        cn_max=par.get("charge.eeq.cn_max"),
        cn_kcn=par.get("charge.eeq.kcn"),
        **dd,
    )
