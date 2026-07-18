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
Run tests for energy contribution from short-ranged bond correction.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN0_XTB as par
from dxtb import IndexHelper
from dxtb._src.components.classicals import new_shortranged
from dxtb._src.typing import DD

from ...conftest import DEVICE
from .samples import samples

sample_list = [
    "H2",
    "H2O",
    "SiH4",
    "ZnOOH-",
    "MB16_43_01",
    "MB16_43_02",
    "LYS_xao",
]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    """Test the short-ranged bond correction for molecules."""

    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]

    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["gfn0"].to(**dd)

    srb = new_shortranged(torch.unique(numbers), par, **dd)
    assert srb is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = srb.get_cache(numbers, ihelp)
    energy = srb.get_energy(positions, cache)

    assert pytest.approx(ref.cpu(), abs=tol) == torch.sum(energy).cpu()
