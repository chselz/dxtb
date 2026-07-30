# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""Pure coordination-number helpers shared by GFN0 classical terms."""

from __future__ import annotations

import torch
from tad_mctc.data import en, radii
from tad_mctc.ncoord import coordination_number, erf_count

from dxtb._src.typing import DD, CountingFunction, Tensor

__all__ = [
    "GFN0_ANGSTROM_PER_BOHR",
    "GFN0_D4_GA",
    "GFN0_D4_GC",
    "GFN0_D4_WF",
    "legacy_d3_radii",
    "gfn0_d4_coordination_number",
]


MODERN_ANGSTROM_PER_BOHR = 0.52917721067
GFN0_ANGSTROM_PER_BOHR = 0.52917726
"""Legacy length-conversion constant used by the GFN0 reference."""

GFN0_D4_GA = 3.0
GFN0_D4_GC = 2.0
GFN0_D4_WF = 6.0
"""Reference-charge model constants used by GFN0 D4."""

_D4_K4 = 4.10451
_D4_K5 = 19.08857
_D4_K6 = 2.0 * 11.28174**2
_D4_CN_CUTOFF = 40.0


def legacy_d3_radii(
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return D3 covalent radii converted with the legacy GFN0 constant."""
    scale = MODERN_ANGSTROM_PER_BOHR / GFN0_ANGSTROM_PER_BOHR
    return radii.COV_D3(device=device, dtype=dtype) * scale


def gfn0_d4_coordination_number(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction | None = None,
) -> Tensor:
    """Evaluate the covalency-weighted GFN0 D4 coordination number."""
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    rcov = legacy_d3_radii(**dd)[numbers]
    pauling = en.PAULING(**dd)[numbers]

    difference = torch.abs(pauling.unsqueeze(-2) - pauling.unsqueeze(-1))
    pair_weight = _D4_K4 * torch.exp(-((difference + _D4_K5) ** 2) / _D4_K6)

    return coordination_number(
        numbers,
        positions,
        counting_function=(
            erf_count if counting_function is None else counting_function
        ),
        rcov=rcov,
        cutoff=_D4_CN_CUTOFF,
        pair_weight=pair_weight,
    )
