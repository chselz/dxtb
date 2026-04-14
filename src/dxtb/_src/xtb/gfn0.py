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
xTB Hamiltonians: GFN0-xTB
==========================

The GFN0-xTB Hamiltonian.
"""

from __future__ import annotations

from functools import partial

import torch
from tad_mctc import storch

from dxtb import IndexHelper
from dxtb._src.components.interactions import Potential
from dxtb._src.param.base import Param
from dxtb._src.param.module import ParameterModule, ParamModule
from dxtb._src.typing import Any, Tensor, override

from .base import PAD, BaseHamiltonian

__all__ = ["GFN0Hamiltonian"]


class GFN0Hamiltonian(BaseHamiltonian):
    """
    The GFN0-xTB Hamiltonian
    """

    def __init__(
        self,
        numbers: Tensor,
        par: Param | ParamModule,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(numbers, par, ihelp, device, dtype)

    def _get_hscale(self, par: ParameterModule) -> Tensor:
        """
        Obtain the off-site scaling factor for the Hamiltonian.

        Parameters
        ----------
        par : ParamModule
            Representation of an extended tight-binding model.

        Returns
        -------
        Tensor
            Off-site scaling factor for the Hamiltonian.
        """
        if par.is_none("hamiltonian"):
            raise RuntimeError("No Hamiltonian specified.")

        ushells = self.ihelp.unique_angular

        angular2label = {
            0: "s",
            1: "p",
            2: "d",
            3: "f",
            4: "g",
        }
        angular_labels = [angular2label.get(int(ang), PAD) for ang in ushells]

        return ksh

    @override
    def _get_elem_valence(self, par: ParamModule) -> Tensor:
        """
        Obtain a mask for valence and non-valence shells. This is only required
        for GFN0-xTB's second hydrogen s-function.

        Parameters
        ----------
        par : ParamModule
            Representation of an extended tight-binding model.

        Returns
        -------
        Tensor
            Mask indicating valence shells for each unique species.
        """
        return par.get_elem_valence(self.unique, pad_val=PAD)

    def get_gradient(
        self,
        positions: Tensor,
        overlap: Tensor,
        doverlap: Tensor,
        pmat: Tensor,
        wmat: Tensor,
        pot: Potential,
        cn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("GFN2 not implemented yet.")
