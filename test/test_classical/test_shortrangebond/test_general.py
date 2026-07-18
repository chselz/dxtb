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
General short-ranged bond correction tests.
"""

import pytest
import torch
from tad_mctc.convert import str_to_device

from dxtb import GFN0_XTB, GFN1_XTB
from dxtb._src.components.classicals import new_shortranged


def test_none_for_non_gfn0() -> None:
    """SRB is only available for GFN0."""
    unique = torch.tensor([6, 7])
    assert new_shortranged(unique, GFN1_XTB) is None


# def test_fail_if_section_missing() -> None:
#     """Factory must fail for GFN0 if the global section is missing."""
#     unique = torch.tensor([6, 7])
#     par = GFN0_XTB.model_copy(deep=True)
#     par.shortrangebond = None

#     with pytest.raises(ValueError):
#         new_shortranged(unique, par)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    """Test changing the `dtype` of the SRB class."""
    cls = new_shortranged(torch.tensor([6, 7]), GFN0_XTB)
    assert cls is not None

    cls = cls.type(dtype)
    assert cls.dtype == dtype


def test_change_type_fail() -> None:
    """Test failure upon changing `dtype` incorrectly."""
    cls = new_shortranged(torch.tensor([6, 7]), GFN0_XTB)
    assert cls is not None

    with pytest.raises(AttributeError):
        cls.dtype = torch.float64

    with pytest.raises(ValueError):
        cls.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    """Test changing the `device` of the SRB class."""
    device = str_to_device(device_str)
    cls = new_shortranged(torch.tensor([6, 7]), GFN0_XTB)
    assert cls is not None

    cls = cls.to(device)
    assert cls.device == device


def test_change_device_fail() -> None:
    """Test failure upon changing `device` incorrectly."""
    cls = new_shortranged(torch.tensor([6, 7]), GFN0_XTB)
    assert cls is not None

    with pytest.raises(AttributeError):
        cls.device = "cpu"


def test_fail_requires_ihelp() -> None:
    """SRB cache construction requires an IndexHelper."""
    numbers = torch.tensor([6, 7])
    cls = new_shortranged(torch.unique(numbers), GFN0_XTB)
    assert cls is not None

    with pytest.raises(ValueError):
        cls.get_cache(numbers=numbers, ihelp=None)
