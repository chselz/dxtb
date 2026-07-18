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
"""Run behavioral tests for the GFN0 short-ranged bond correction."""

from __future__ import annotations

import pytest
import torch

from dxtb import IndexHelper
from dxtb._src.components.classicals import new_shortranged
from dxtb._src.param.gfn0 import GFN0_XTB
from dxtb._src.typing import DD

from ...conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_pair_scope_mask(dtype: torch.dtype) -> None:
    """SRB mask must follow exact gfn0 scope rules."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([6, 7, 8, 6, 1], device=DEVICE)

    srb = new_shortranged(torch.unique(numbers), GFN0_XTB, **dd)
    assert srb is not None

    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    cache = srb.get_cache(numbers, ihelp)

    pair_mask = torch.ones(
        (numbers.shape[-1], numbers.shape[-1]), dtype=torch.bool
    )
    pair_mask.fill_diagonal_(False)

    srb_atoms = (numbers >= 5) & (numbers <= 9)
    expected = (
        pair_mask
        & (srb_atoms.unsqueeze(-1) & srb_atoms.unsqueeze(-2))
        & (numbers.unsqueeze(-1) != numbers.unsqueeze(-2))
    )

    assert torch.equal(cache.mask, expected)


def test_energy_zero_outside_scope() -> None:
    """Pairs outside [5,9] must not contribute to SRB."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = torch.tensor([1, 8], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]],
        **dd,
    )

    srb = new_shortranged(torch.unique(numbers), GFN0_XTB, **dd)
    assert srb is not None

    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    cache = srb.get_cache(numbers, ihelp)
    emat = srb.get_energy(positions, cache, atom_resolved=False)

    assert torch.count_nonzero(emat).item() == 0


def test_energy_zero_beyond_cutoff() -> None:
    """Pairs beyond the configured SRB cutoff must be zeroed."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = torch.tensor([6, 7], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        **dd,
    )

    srb = new_shortranged(torch.unique(numbers), GFN0_XTB, cutoff2=1.0, **dd)
    assert srb is not None

    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    cache = srb.get_cache(numbers, ihelp)
    emat = srb.get_energy(positions, cache, atom_resolved=False)

    assert torch.count_nonzero(emat).item() == 0


def test_energy_nonzero_for_allowed_pair() -> None:
    """Heteronuclear pairs in [5,9] should contribute a negative SRB term."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = torch.tensor([6, 7], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]],
        **dd,
    )

    srb = new_shortranged(torch.unique(numbers), GFN0_XTB, **dd)
    assert srb is not None

    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    cache = srb.get_cache(numbers, ihelp)

    emat = srb.get_energy(positions, cache, atom_resolved=False)
    eatom = srb.get_energy(positions, cache, atom_resolved=True)

    assert emat[0, 1] < 0.0
    assert emat[1, 0] < 0.0
    assert pytest.approx(emat[0, 1].item()) == emat[1, 0].item()
    assert torch.count_nonzero(torch.diag(emat)).item() == 0
    assert torch.allclose(eatom, 0.5 * emat.sum(-1))


def tes_energy_zero_for_homonuclear_pairs() -> None:
    """Homonuclear pairs in [5,9] should not contribute to SRB."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = torch.tensor([6, 6], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]],
        **dd,
    )

    srb = new_shortranged(torch.unique(numbers), GFN0_XTB, **dd)
    assert srb is not None

    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    cache = srb.get_cache(numbers, ihelp)

    emat = srb.get_energy(positions, cache, atom_resolved=False)

    assert torch.count_nonzero(emat).item() == 0


def test_autograd_positions_smoke() -> None:
    """SRB energy should be differentiable with respect to positions."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = torch.tensor([6, 7], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]],
        **dd,
        requires_grad=True,
    )

    srb = new_shortranged(torch.unique(numbers), GFN0_XTB, **dd)
    assert srb is not None

    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    cache = srb.get_cache(numbers, ihelp)

    energy = 0.5 * srb.get_energy(positions, cache, atom_resolved=False).sum()
    (grad,) = torch.autograd.grad(energy, positions)

    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0.0
