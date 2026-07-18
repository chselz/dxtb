# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Short-ranged Bond Correction: Class
===================================

This module implements the short-ranged bond correction class. The
:class:`dxtb.components.Shortranged` class is constructed similar to the
:class:`dxtb.components.Repulsion` class.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.convert import any_to_tensor
from tad_mctc.ncoord import cn_d3, erf_count
from tad_mctc.typing import Any, Tensor, override

from dxtb import IndexHelper

from ..base import Classical, ClassicalCache

__all__ = ["ShortrangedCache", "Shortranged", "LABEL_SHORTRANGED"]


LABEL_SHORTRANGED = "Shortranged"
"""
Label for the :class:`.Shortranged` component, coinciding with the class name.
"""


class ShortrangedCache(ClassicalCache):
    """Cache for the short-ranged bond correction."""

    numbers: Tensor
    """Atomic numbers used for CN evaluation and masking."""

    mask: Tensor
    """Pair mask for SRB interactions."""

    srbr0: Tensor
    """Reference SRB radii."""

    srbcn: Tensor
    """CN-dependent SRB radius coefficients."""

    srben: Tensor
    """Element-specific SRB electronegativities for ``f_f``."""

    pauling_en: Tensor
    """Pauling electronegativities for the steepness scaling."""

    k1: Tensor
    """Linear EN polynomial prefactor for ``f_f``."""

    k2: Tensor
    """Quadratic EN polynomial prefactor for ``f_f``."""

    __slots__ = [
        "numbers",
        "mask",
        "srbr0",
        "srbcn",
        "srben",
        "pauling_en",
        "k1",
        "k2",
    ]

    def __init__(
        self,
        numbers: Tensor,
        mask: Tensor,
        srbr0: Tensor,
        srbcn: Tensor,
        srben: Tensor,
        pauling_en: Tensor,
        k1: Tensor,
        k2: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is None else srbr0.device,
            dtype=dtype if dtype is None else srbr0.dtype,
        )
        self.numbers = numbers
        self.mask = mask
        self.srbr0 = srbr0
        self.srbcn = srbcn
        self.srben = srben
        self.pauling_en = pauling_en
        self.k1 = k1
        self.k2 = k2


class Shortranged(Classical):
    """Representation of the short-ranged bond correction."""

    srbr0: Tensor
    """Element-specific reference radii for unique species."""

    srbcn: Tensor
    """Element-specific CN scaling of the SRB radius."""

    srben: Tensor
    """Element-specific EN values for the SRB distance scaling polynomial."""

    pauling_en: Tensor
    """Element-specific Pauling EN values used in the steepness prefactor."""

    rows: Tensor
    """EN polynomial coefficients grouped by periodic-row buckets."""

    shift: Tensor
    """Global additive shift for reference bond radii."""

    prefactor: Tensor
    """Global SRB energy prefactor."""

    steepness: Tensor
    """Global steepness factor for the Gaussian damping."""

    enscale: Tensor
    """Global EN scaling factor for steepness."""

    cutoff2: Tensor
    """Squared real-space cutoff used for SRB pair selection."""

    cnmax: Tensor
    """Upper clipping value for CN in SRB."""

    __slots__ = [
        "srbr0",
        "srbcn",
        "srben",
        "pauling_en",
        "rows",
        "shift",
        "prefactor",
        "steepness",
        "enscale",
        "cutoff2",
        "cnmax",
    ]

    def __init__(
        self,
        srbr0: Tensor,
        srbcn: Tensor,
        srben: Tensor,
        pauling_en: Tensor,
        rows: Tensor,
        shift: Tensor | float | int,
        prefactor: Tensor | float | int,
        steepness: Tensor | float | int,
        enscale: Tensor | float | int,
        cutoff2: Tensor | float | int,
        cnmax: Tensor | float | int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)

        self.srbr0 = srbr0.to(**self.dd)
        self.srbcn = srbcn.to(**self.dd)
        self.srben = srben.to(**self.dd)
        self.pauling_en = pauling_en.to(**self.dd)
        self.rows = rows.to(**self.dd)
        self.shift = any_to_tensor(shift, **self.dd)
        self.prefactor = any_to_tensor(prefactor, **self.dd)
        self.steepness = any_to_tensor(steepness, **self.dd)
        self.enscale = any_to_tensor(enscale, **self.dd)
        self.cutoff2 = any_to_tensor(cutoff2, **self.dd)
        self.cnmax = any_to_tensor(cnmax, **self.dd)

    @staticmethod
    def _row_index(numbers: Tensor) -> Tensor:
        """
        Map atomic numbers to SRB row groups used in gfn0.

        Buckets:
        - row 1: ``Z <= 2``
        - row 2: ``2 < Z <= 10``
        - row 3: ``10 < Z <= 18``
        - row 4: ``Z > 18``
        """
        return torch.where(
            numbers <= 2,
            numbers.new_zeros(numbers.shape),
            torch.where(
                numbers <= 10,
                numbers.new_ones(numbers.shape),
                torch.where(
                    numbers <= 18,
                    numbers.new_full(numbers.shape, 2),
                    numbers.new_full(numbers.shape, 3),
                ),
            ),
        )

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **_: Any
    ) -> ShortrangedCache:
        """
        Store geometry-independent variables for SRB energy calculation.
        """
        if ihelp is None:
            raise ValueError("IndexHelper must be passed for short-ranged.")

        cachvars = (numbers.detach().clone(),)

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, ShortrangedCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}.Cache'. "
                    "This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        self._cachevars = cachvars

        srbr0 = ihelp.spread_uspecies_to_atom(self.srbr0)
        srbcn = ihelp.spread_uspecies_to_atom(self.srbcn)
        srben = ihelp.spread_uspecies_to_atom(self.srben)
        pauling_en = ihelp.spread_uspecies_to_atom(self.pauling_en)

        # Exact gfn0 SRB scope: both atoms in [5, 9] and hetero-only.
        pair_mask = real_pairs(numbers, mask_diagonal=True)
        srb_atoms = (numbers >= 5) & (numbers <= 9)
        pair_mask = (
            pair_mask
            & (srb_atoms.unsqueeze(-1) & srb_atoms.unsqueeze(-2))
            & (numbers.unsqueeze(-1) != numbers.unsqueeze(-2))
        )

        rows = self._row_index(numbers)
        prow = self.rows[rows]
        p1 = prow[..., 0]
        p2 = prow[..., 1]

        scale = 0.005
        k1 = scale * (p1.unsqueeze(-1) + p1.unsqueeze(-2))
        k2 = scale * (p2.unsqueeze(-1) + p2.unsqueeze(-2))

        self.cache = ShortrangedCache(
            numbers,
            pair_mask,
            srbr0,
            srbcn,
            srben,
            pauling_en,
            k1,
            k2,
        )
        return self.cache

    @override
    def get_energy(
        self, positions: Tensor, cache: ShortrangedCache, **kwargs: Any
    ) -> Tensor:
        """
        Get short-ranged bond correction energy.
        """
        if not isinstance(cache, ShortrangedCache):
            raise TypeError(
                f"Cache in {self.label} is not of type 'ShortrangedCache'."
            )

        zero = positions.new_zeros(())
        cn = cn_d3(cache.numbers, positions, counting_function=erf_count)
        cn = torch.clamp(cn, max=self.cnmax)

        ra = cache.srbr0 + cache.srbcn * cn + self.shift

        den_srb = torch.abs(
            cache.srben.unsqueeze(-1) - cache.srben.unsqueeze(-2)
        )
        ff = 1.0 - cache.k1 * den_srb - cache.k2 * den_srb.pow(2)
        rab0 = (ra.unsqueeze(-1) + ra.unsqueeze(-2)) * ff

        den_pauling = cache.pauling_en.unsqueeze(
            -1
        ) - cache.pauling_en.unsqueeze(-2)
        pre = self.steepness * (1.0 + self.enscale * den_pauling.pow(2))

        distances = storch.cdist(positions, positions, p=2)
        dr = distances - rab0

        e = self.prefactor * torch.exp(-pre * dr.pow(2))
        mask = cache.mask & (distances.pow(2) < self.cutoff2)
        e = torch.where(mask, e, zero)

        if kwargs.get("atom_resolved", True) is True:
            print("e sum:", torch.sum(e).item(), "expected:", ref.cpu().item())
            return 0.5 * torch.sum(e, dim=-1)
        return e
