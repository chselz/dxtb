# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""Tests for the one-shot GFN0 electronic solver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import IndexHelper, labels
from dxtb._src.components.interactions import (
    Interaction,
    InteractionList,
)
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.integral.factory import new_hcore
from dxtb._src.integral.wrappers import overlap
from dxtb._src.param import GFN0_XTB
from dxtb._src.scf.iterator import solve
from dxtb._src.scf.pure.conversions import diagonalize
from dxtb._src.scf.result import SCFResult
from dxtb._src.typing import Tensor
from dxtb._src.xtb.gfn0 import GFN0Hamiltonian
from dxtb.config import ConfigSCF

REFS = Path(__file__).parents[1] / "test_gfn0" / "refs"


def load_fixture(name: str) -> dict[str, Any]:
    """Load a frozen GFN0 reference fixture."""
    return json.loads((REFS / f"{name}.json").read_text())


def build_problem(
    name: str,
) -> tuple[
    dict[str, Any],
    Tensor,
    Tensor,
    IndexHelper,
    GFN0Hamiltonian,
    IntegralMatrices,
]:
    """Build H0/S and indexing data for a frozen system."""
    fixture = load_fixture(name)
    numbers = torch.tensor(fixture["numbers"])
    positions = torch.tensor(fixture["positions"], dtype=torch.float64)
    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    hcore = new_hcore(numbers, GFN0_XTB, ihelp, dtype=torch.float64)
    assert isinstance(hcore, GFN0Hamiltonian)
    ovlp = overlap(
        numbers,
        positions,
        GFN0_XTB,
        driver=labels.INTDRIVER_ANALYTICAL,
    )
    h0 = hcore.build(
        positions, ovlp, charge=fixture["metadata"]["total_charge"]
    )
    integrals = IntegralMatrices(_hcore=h0, _overlap=ovlp, dtype=torch.float64)
    return fixture, numbers, positions, ihelp, hcore, integrals


def run_problem(
    name: str,
    *,
    config: ConfigSCF | None = None,
    interactions: InteractionList | None = None,
) -> tuple[dict[str, Any], SCFResult]:
    """Run the non-self-consistent solver for one frozen system."""
    fixture, numbers, positions, ihelp, hcore, integrals = build_problem(name)
    if config is None:
        config = ConfigSCF(method=labels.GFN0_XTB, dtype=torch.float64)
    if interactions is None:
        interactions = InteractionList()
    cache = interactions.get_cache(numbers, positions, ihelp)
    charge = torch.tensor(
        [fixture["metadata"]["total_charge"]], dtype=torch.float64
    )
    result = solve(
        numbers,
        positions,
        charge,
        None,
        interactions,
        cache,
        ihelp,
        config,
        integrals,
        hcore.refocc,
    )
    return fixture, result


def test_requires_iterations_is_derived() -> None:
    """Select one-shot execution only from the configured physical method."""
    gfn0 = ConfigSCF(method=labels.GFN0_XTB)
    assert gfn0.requires_iterations is False
    assert ConfigSCF(method=labels.GFN1_XTB).requires_iterations is True
    assert ConfigSCF(method=labels.GFN2_XTB).requires_iterations is True

    with pytest.raises(AttributeError):
        gfn0.requires_iterations = True  # type: ignore[misc]


@pytest.mark.parametrize("name", ["H2", "H2O", "LiH", "ZnOOH-"])
def test_complete_result_and_frozen_eht(name: str) -> None:
    """Populate every result field and reproduce the frozen band energy."""
    fixture, result = run_problem(name)

    assert set(result) == {
        "charges",
        "coefficients",
        "density",
        "emo",
        "energy",
        "fenergy",
        "hamiltonian",
        "occupation",
        "potential",
        "iterations",
    }
    assert result["iterations"] == 0
    assert result["energy"].sum().item() == pytest.approx(
        fixture["energies"]["eht"], abs=5.0e-7
    )
    assert result["charges"].mono.sum().item() == pytest.approx(
        fixture["metadata"]["total_charge"], abs=1.0e-10
    )
    assert torch.count_nonzero(result["potential"].mono).item() == 0
    assert torch.isfinite(result["coefficients"]).all()
    assert torch.isfinite(result["density"]).all()
    assert torch.isfinite(result["emo"]).all()
    assert torch.isfinite(result["occupation"]).all()

    for orbital in fixture["orbitals"]:
        index = orbital["index"] - 1
        assert result["emo"][index].item() == pytest.approx(
            orbital["energy"], abs=1.0e-7
        )
        assert result["occupation"].sum(-2)[index].item() == pytest.approx(
            orbital["occupation"], abs=1.0e-12
        )


def test_exactly_one_diagonalization_and_no_iteration_machinery() -> None:
    """Dispatch before guesses, mixers, and equilibrium/root solvers."""
    config = ConfigSCF(
        method=labels.GFN0_XTB,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        scp_mode=labels.SCP_MODE_CHARGE,
        maxiter=999,
        dtype=torch.float64,
    )

    with (
        patch(
            "dxtb._src.scf.pure.conversions.diagonalize", wraps=diagonalize
        ) as diagonalize_mock,
        patch(
            "dxtb._src.scf.iterator.get_guess",
            side_effect=AssertionError("SCF guess must not be called"),
        ),
        patch(
            "dxtb._src.exlibs.xitorch.optimize.equilibrium",
            side_effect=AssertionError("root solver must not be called"),
        ),
        patch(
            "dxtb._src.scf.mixer.Simple.iter",
            side_effect=AssertionError("mixer must not be called"),
        ),
    ):
        _, result = run_problem("H2O", config=config)

    assert result["iterations"] == 0
    assert diagonalize_mock.call_count == 1


@pytest.mark.parametrize(
    ("maxiter", "mixer", "scp_mode", "scf_mode"),
    [
        (0, labels.MIXER_LINEAR, labels.SCP_MODE_CHARGE, labels.SCF_MODE_FULL),
        (
            1,
            labels.MIXER_ANDERSON,
            labels.SCP_MODE_POTENTIAL,
            labels.SCF_MODE_IMPLICIT,
        ),
        (
            99,
            labels.MIXER_BROYDEN,
            labels.SCP_MODE_FOCK,
            labels.SCF_MODE_EXPERIMENTAL,
        ),
        (
            999,
            labels.MIXER_LINEAR,
            labels.SCP_MODE_FOCK,
            labels.SCF_MODE_IMPLICIT_NON_PURE,
        ),
    ],
)
def test_iterative_options_are_inert(
    maxiter: int, mixer: int, scp_mode: int, scf_mode: int
) -> None:
    """Ignore iterative strategy controls for the physical GFN0 method."""
    config = ConfigSCF(
        method=labels.GFN0_XTB,
        maxiter=maxiter,
        mixer=mixer,
        scp_mode=scp_mode,
        scf_mode=scf_mode,
        dtype=torch.float64,
    )
    fixture, result = run_problem("H2O", config=config)
    assert result["iterations"] == 0
    assert result["energy"].sum().item() == pytest.approx(
        fixture["energies"]["eht"], abs=5.0e-7
    )


def test_charge_dependent_interaction_is_rejected() -> None:
    """Reject an arbitrary one-pass interaction correction for GFN0."""
    interactions = InteractionList(Interaction(dtype=torch.float64))
    with pytest.raises(
        ValueError, match="does not support charge-dependent Interaction"
    ):
        run_problem("H2", interactions=interactions)


def test_heterogeneous_batch_matches_separate_solves() -> None:
    """Match packed H2/H2O results to separate one-shot calculations."""
    names = ("H2", "H2O")
    fixtures = [load_fixture(name) for name in names]
    number_list = [torch.tensor(fixture["numbers"]) for fixture in fixtures]
    position_list = [
        torch.tensor(fixture["positions"], dtype=torch.float64)
        for fixture in fixtures
    ]
    numbers = pack(number_list)
    positions = pack(position_list)
    charges = torch.tensor(
        [[fixture["metadata"]["total_charge"]] for fixture in fixtures],
        dtype=torch.float64,
    )

    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    hcore = new_hcore(numbers, GFN0_XTB, ihelp, dtype=torch.float64)
    assert isinstance(hcore, GFN0Hamiltonian)
    ovlp = overlap(
        numbers,
        positions,
        GFN0_XTB,
        driver=labels.INTDRIVER_ANALYTICAL,
    )
    h0 = hcore.build(positions, ovlp, charge=charges)
    integrals = IntegralMatrices(_hcore=h0, _overlap=ovlp, dtype=torch.float64)
    interactions = InteractionList()
    result = solve(
        numbers,
        positions,
        charges,
        None,
        interactions,
        interactions.get_cache(numbers, positions, ihelp),
        ihelp,
        ConfigSCF(
            method=labels.GFN0_XTB,
            batch_mode=1,
            dtype=torch.float64,
        ),
        integrals,
        hcore.refocc,
    )

    separate = [run_problem(name)[1] for name in names]
    assert result["iterations"] == 0
    assert torch.allclose(
        result["energy"].sum(-1),
        torch.stack([item["energy"].sum() for item in separate]),
        atol=1.0e-10,
        rtol=0.0,
    )
    # Generalized diagonalization sorts the packed padding eigenvalues (zero)
    # between negative and positive physical virtual levels. Compare the four
    # real H2 levels while excluding those padding zeros.
    packed_h2_emo = torch.cat((result["emo"][0, :3], result["emo"][0, -1:]))
    assert torch.allclose(
        packed_h2_emo,
        separate[0]["emo"],
        atol=1.0e-10,
        rtol=0.0,
    )
    assert torch.allclose(
        result["emo"][1, : separate[1]["emo"].shape[-1]],
        separate[1]["emo"],
        atol=1.0e-10,
        rtol=0.0,
    )
    assert torch.count_nonzero(result["potential"].mono).item() == 0
