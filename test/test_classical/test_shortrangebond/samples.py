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
Data for testing repulsion taken from https://github.com/grimme-lab/mstore.
"""

from __future__ import annotations

import torch
from tad_mctc.data.molecules import merge_nested_dicts, mols

from dxtb._src.typing import Molecule, Tensor, TypedDict


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    gfn0: Tensor
    """Referenece values for GFN0-xTB"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "H2": {
        "gfn0": torch.tensor(0.000000000000),
    },
    "H2O": {
        "gfn0": torch.tensor(0.000000000000),
    },
    "SiH4": {
        "gfn0": torch.tensor(0.000000000000),
    },
    "ZnOOH-": {
        "gfn0": torch.tensor(0.000000000000),
    },
    "MB16_43_01": {
        "gfn0": torch.tensor(-0.026453365313),
    },
    "MB16_43_02": {
        "gfn0": torch.tensor(-0.000000000000),
    },
    "LYS_xao": {
        "gfn0": torch.tensor(-0.088162441935),
    },
}

samples: dict[str, Record] = merge_nested_dicts(mols, refs)
