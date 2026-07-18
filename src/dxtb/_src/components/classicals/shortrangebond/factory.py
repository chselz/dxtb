# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Short-ranged Bond Correction: Factory
================================

A factory function to create instances of the :class:`dxtb.components.Shortranged`
class.
"""

from __future__ import annotations

import torch
from tad_mctc.convert import any_to_tensor

from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import DD, Tensor, get_default_dtype

from .srb import Shortranged

__all__ = ["new_shortranged"]


def new_shortranged(
    unique: Tensor,
    par: Param | ParamModule,
    cutoff2: Tensor | float | int | None = None,
    cnmax: Tensor | float | int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Shortranged | None:
    """
    Create a new instance of the short-ranged bond correction.

    Parameters
    ----------
    unique : Tensor
        Unique elements in the system (shape: ``(nunique,)``).
    par : Param | ParamModule
        Representation of an extended tight-binding model.
    cutoff2 : Tensor | float | int | None, optional
        Squared real-space cutoff for SRB interactions. If ``None``, the value
        is read from ``[shortrangebond].cutoff2`` or defaults to ``200.0``.
    cnmax : Tensor | float | int | None, optional
        Upper clipping value for the coordination number used in SRB. If
        ``None``, the value is read from ``[shortrangebond].cnmax`` or defaults
        to ``8.0``.

    Returns
    -------
    Shortranged | None
        Instance of :class:`.Shortranged` or ``None`` for non-GFN0 models.

    Raises
    ------
    ValueError
        If GFN0 SRB parameters are missing.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    # compatibility with previous version based on `Param`
    if not isinstance(par, ParamModule):
        par = ParamModule(par, **dd)

    if "shortrangebond" not in par or par.is_none("shortrangebond"):
        return None

    section = par.get("shortrangebond")

    shift = par.get("shortrangebond.shift")
    prefactor = par.get("shortrangebond.prefactor")
    steepness = par.get("shortrangebond.steepness")
    enscale = par.get("shortrangebond.enscale")

    rows = torch.stack(
        [
            par.get("shortrangebond.row1"),
            par.get("shortrangebond.row2"),
            par.get("shortrangebond.row3"),
            par.get("shortrangebond.row4"),
        ],
        dim=0,
    )

    if cutoff2 is None:
        cutoff2 = (
            par.get("shortrangebond.cutoff2")
            if "cutoff2" in section
            else any_to_tensor(200.0, **dd)
        )
    cutoff2 = any_to_tensor(cutoff2, **dd)

    if cnmax is None:
        cnmax = (
            par.get("shortrangebond.cnmax")
            if "cnmax" in section
            else any_to_tensor(8.0, **dd)
        )
    cnmax = any_to_tensor(cnmax, **dd)

    try:
        srbr0 = par.get_elem_param(unique, "srbr0", pad_val=0)
        srbcn = par.get_elem_param(unique, "srbcn", pad_val=0)
        srben = par.get_elem_param(unique, "srben", pad_val=0)
    except KeyError as exc:
        raise ValueError(
            "GFN0 SRB element parameters are missing. Expected element fields "
            "'srbr0', 'srbcn', and 'srben'."
        ) from exc

    pauling_en = par.get_elem_param(unique, "en", pad_val=0)

    return Shortranged(
        srbr0=srbr0,
        srbcn=srbcn,
        srben=srben,
        pauling_en=pauling_en,
        rows=rows,
        shift=shift,
        prefactor=prefactor,
        steepness=steepness,
        enscale=enscale,
        cutoff2=cutoff2,
        cnmax=cnmax,
        **dd,
    )
