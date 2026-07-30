# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""Factory for the GFN0-xTB isotropic electrostatic energy."""

from __future__ import annotations

import torch

from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import DD, Tensor, get_default_dtype

from ..gfn0 import legacy_d3_radii
from .ies import IES

__all__ = ["new_ies"]


def new_ies(
    numbers: Tensor,
    par: Param | ParamModule,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IES | None:
    """
    Create the GFN0-xTB isotropic electrostatic classical contribution.

    Parameterizations without a ``charge.eeq`` block do not select this
    component and return ``None``.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    if not isinstance(par, ParamModule):
        par = ParamModule(par, **dd)

    if "charge" not in par or par.is_none("charge.eeq"):
        return None

    cn = par.get("charge.eeq.cn")
    if cn != "erf":
        raise ValueError(
            "The GFN0 IES component only supports erf coordination numbers."
        )

    max_element = int(numbers.max().item()) if numbers.numel() > 0 else 0
    elements = torch.arange(
        max_element + 1,
        dtype=numbers.dtype,
        device=numbers.device,
    )

    return IES(
        chi=par.get_elem_param(elements, "eeq_chi", pad_val=0),
        eeq_kcn=par.get_elem_param(elements, "eeq_kcn", pad_val=0),
        eta=par.get_elem_param(elements, "eeq_eta", pad_val=0),
        rad=par.get_elem_param(elements, "eeq_rad", pad_val=0),
        rcov=legacy_d3_radii(**dd),
        cutoff=par.get("charge.eeq.cutoff"),
        cn_max=par.get("charge.eeq.cn_max"),
        cn_kcn=par.get("charge.eeq.kcn"),
        **dd,
    )
