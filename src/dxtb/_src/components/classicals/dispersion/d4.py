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
Dispersion: D4
==============

DFT-D4 dispersion model.
"""

from __future__ import annotations

from typing import Any

import tad_dftd4 as d4
import torch
from tad_dftd4.dispersion import TwoBodyTerm
from tad_dftd4.dispersion.base import Disp
from tad_mctc.data import radii
from tad_mctc.ncoord import coordination_number, erf_count
from tad_mctc.typing import CountingFunction, Tensor, override
from tad_multicharge.model.eeq import EEQModel

from dxtb import IndexHelper

from ..base import ClassicalCache, ComponentCache
from ..gfn0 import (
    GFN0_D4_GA,
    GFN0_D4_GC,
    GFN0_D4_WF,
    gfn0_d4_coordination_number,
    legacy_d3_radii,
)
from .base import Dispersion

__all__ = ["DispersionD4", "DispersionD4Cache", "DispersionD4GFN0"]


class DispersionD4Cache(ClassicalCache):
    """
    Cache for the dispersion settings.

    Note
    ----
    The dispersion parameters (a1, a2, ...) are given in the dispersion
    class constructor.
    """

    __slots__ = [
        "q",
        "model",
        "rcov",
        "r4r2",
        "cutoff",
        "counting_function",
        "damping_function",
        "eeq_model",
    ]

    def __init__(
        self,
        q: Tensor | None,
        model: d4.model.D4Model,
        rcov: Tensor,
        r4r2: Tensor,
        cutoff: d4.cutoff.Cutoff,
        counting_function: CountingFunction,
        damping_function: d4.damping.Damping,
        eeq_model: EEQModel | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.q = q
        self.model = model
        self.rcov = rcov
        self.r4r2 = r4r2
        self.cutoff = cutoff
        self.counting_function = counting_function
        self.damping_function = damping_function
        self.eeq_model = eeq_model


class DispersionD4(Dispersion):
    """
    Representation of the DFT-D4 dispersion correction (:class:`.DispersionD4`).
    """

    # pylint: disable=unused-argument
    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **kwargs: Any
    ) -> DispersionD4Cache:
        """
        Obtain cache for storage of settings.

        Settings can be passed as `kwargs`. The available optional parameters
        are the same as in `tad_dftd4.dftd4`, i.e., "model", "rcov", "r4r2",
        "cutoff", "counting_function", and "damping_function".

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        Returns
        -------
        DispersionD4Cache
            Cache for the D4 dispersion.
        """
        cachvars = (numbers.detach().clone(),)

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, DispersionD4Cache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        self._cachevars = cachvars

        model = kwargs.pop("model", None)
        if model is not None and not isinstance(model, d4.model.D4Model):
            raise TypeError("D4: Model is not of type 'd4.model.D4Model'.")
        if model is None:
            model = d4.model.D4Model(
                numbers, ref_charges=self.ref_charges, **self.dd
            )
        else:
            model = model.type(self.dtype).to(self.device)

        rcov = kwargs.pop("rcov", None)
        if rcov is not None and not isinstance(rcov, Tensor):
            raise TypeError("D4: 'rcov' is not of type 'Tensor'.")
        if rcov is None:
            rcov = radii.COV_D3(**self.dd)[numbers]
        else:
            rcov = rcov.to(**self.dd)

        r4r2 = kwargs.pop("r4r2", None)
        if r4r2 is not None and not isinstance(r4r2, Tensor):
            raise TypeError("D4: 'r4r2' is not of type 'Tensor'.")
        if r4r2 is None:
            r4r2 = d4.data.R4R2(**self.dd)[numbers]
        else:
            r4r2 = r4r2.to(**self.dd)

        cutoff = kwargs.pop("cutoff", None)
        if cutoff is not None and not isinstance(cutoff, d4.Cutoff):
            raise TypeError("D4: 'cutoff' is not of type 'd4.Cutoff'.")
        if cutoff is None:
            cutoff = d4.Cutoff(**self.dd)
        else:
            cutoff = cutoff.type(self.dtype).to(self.device)

        q = kwargs.pop("q", None)

        cf = kwargs.pop("counting_function", erf_count)
        df = kwargs.pop("damping_function", d4.damping.RationalDamping())

        self.cache = DispersionD4Cache(q, model, rcov, r4r2, cutoff, cf, df)
        return self.cache

    @override
    def get_energy(
        self,
        positions: Tensor,
        cache: ComponentCache,
        q: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Get D4 dispersion energy.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        cache : ComponentCache
            Dispersion cache containing settings.
        q : Tensor | None, optional
            Atomic partial charges. Defaults to ``None`` (EEQ charges).

        Returns
        -------
        Tensor
            Atom-resolved D4 dispersion energy.
        """
        if not isinstance(cache, DispersionD4Cache):
            raise TypeError(
                f"Cache in {self.label} is not of type '{self.label}Cache'."
            )

        charge = kwargs.pop("charge", self.charge)
        if charge is None:
            raise ValueError("Total molecular charge is required for DFT-D4.")

        return d4.dftd4(
            self.numbers,
            positions,
            charge,
            self.param,
            model=cache.model,
            rcov=cache.rcov,
            r4r2=cache.r4r2,
            q=cache.q if q is None else q,
            cutoff=cache.cutoff,
            counting_function=cache.counting_function,
            damping_function=cache.damping_function,
        )


class DispersionD4GFN0(DispersionD4):
    """GFN0 two-body D4 with locally evaluated D4-2019 EEQ charges."""

    __slots__ = ["cn_cutoff", "cn_max", "cn_kcn"]

    def __init__(
        self,
        numbers: Tensor,
        param: d4.Param,
        cn_cutoff: Tensor,
        cn_max: Tensor,
        cn_kcn: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            numbers,
            param,
            charge=None,
            ref_charges="eeq",
            device=device,
            dtype=dtype,
        )
        # Keep the established classical result key.
        self.label = DispersionD4.__name__
        self.cn_cutoff = cn_cutoff.to(**self.dd)
        self.cn_max = cn_max.to(**self.dd)
        self.cn_kcn = cn_kcn.to(**self.dd)

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **kwargs: Any
    ) -> DispersionD4Cache:
        """Build explicit GFN0 D4 model, cutoff, and D4-2019 EEQ data."""
        cachvars = (numbers.detach().clone(),)
        if self.cache_is_latest(cachvars):
            if not isinstance(self.cache, DispersionD4Cache):
                raise TypeError(
                    f"Cache in {self.label} is not of type "
                    "'DispersionD4Cache'."
                )
            return self.cache

        kwargs.setdefault(
            "model",
            d4.model.D4Model(
                numbers,
                ga=GFN0_D4_GA,
                gc=GFN0_D4_GC,
                wf=GFN0_D4_WF,
                ref_charges="eeq",
                **self.dd,
            ),
        )
        kwargs.setdefault(
            "cutoff",
            d4.Cutoff(
                disp2=60.0,
                disp3=40.0,
                cn=40.0,
                cn_eeq=40.0,
                **self.dd,
            ),
        )
        kwargs.setdefault("rcov", legacy_d3_radii(**self.dd)[numbers])

        cache = super().get_cache(numbers, ihelp, **kwargs)
        if cache.eeq_model is None:
            cache.eeq_model = EEQModel.param2019(**self.dd)
        return cache

    def get_coordination_number(
        self, positions: Tensor, cache: DispersionD4Cache
    ) -> Tensor:
        """Evaluate the capped plain-erf CN used for D4 EEQ charges."""
        return coordination_number(
            self.numbers,
            positions,
            counting_function=erf_count,
            rcov=cache.rcov,
            cutoff=self.cn_cutoff,
            cn_max=self.cn_max,
            kcn=self.cn_kcn,
        )

    def get_dispersion(self, cache: DispersionD4Cache) -> Disp:
        """Construct a lower-level D4 evaluator containing only TwoBodyTerm."""
        disp = Disp(
            model=cache.model,
            cn_fn=gfn0_d4_coordination_number,
            cn_fn_kwargs={"counting_function": cache.counting_function},
            **self.dd,
        )
        disp.register(
            TwoBodyTerm(
                damping_fn=cache.damping_function,
                charge_dependent=True,
            )
        )
        return disp

    @override
    def get_energy(
        self,
        positions: Tensor,
        cache: ComponentCache,
        q: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Evaluate atomwise two-body D4 using private D4-2019 EEQ charges."""
        if not isinstance(cache, DispersionD4Cache):
            raise TypeError(
                f"Cache in {self.label} is not of type 'DispersionD4Cache'."
            )
        if q is not None:
            raise ValueError(
                "GFN0 D4 calculates atomic EEQ charges internally."
            )

        charge = kwargs.pop("charge", self.charge)
        if charge is None:
            raise ValueError("Total molecular charge is required for DFT-D4.")
        if cache.eeq_model is None:
            raise RuntimeError("GFN0 D4 cache has no EEQ model.")

        cn = self.get_coordination_number(positions, cache)
        charges = cache.eeq_model.solve(
            self.numbers,
            positions,
            torch.as_tensor(charge, **self.dd),
            cn,
        )
        assert isinstance(charges, Tensor)

        return self.get_dispersion(cache).calculate(
            self.numbers,
            positions,
            charge,
            self.param,
            cutoff=cache.cutoff,
            q=charges,
            rcov=cache.rcov,
            r4r2=cache.r4r2,
        )
