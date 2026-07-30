# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""End-to-end GFN0-xTB validation against frozen xTB references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from tad_mctc.batch import pack

from dxtb._src.calculators.result import Result
from dxtb.calculators import GFN0Calculator

REFS = Path(__file__).parent / "refs"
SYSTEMS = (
    "H2",
    "H2O",
    "LiH",
    "SiH4",
    "MB16_43_01",
    "MB16_43_02",
    "ZnOOH-",
    "LYS_xao",
)
COMPONENTS = {
    "ies": "IES",
    "d4_2body": "DispersionD4",
    "repulsion": "Repulsion",
    "srb": "ShortRangeBond",
}

# The reference output rounds the MB16_43_02 Fermi contribution to zero while
# dxtb retains a 3.02e-7 Eh finite-temperature term and the compensating EHT
# partition. Their sum and the total retain the tight float64 accuracy.
FLOAT64_ELECTRONIC_PARTITION_ATOL = 1.0e-6
FLOAT32_ATOL = 1.0e-5


def load_reference(
    system: str, dtype: torch.dtype
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Load one self-contained GFN0 reference fixture."""
    fixture = json.loads((REFS / f"{system}.json").read_text())
    numbers = torch.tensor(fixture["numbers"])
    positions = torch.tensor(fixture["positions"], dtype=dtype)
    return fixture, numbers, positions


def calculate(
    system: str,
    dtype: torch.dtype,
    *,
    exclude: list[str] | None = None,
) -> tuple[dict[str, Any], GFN0Calculator, Result]:
    """Run the public GFN0 calculator for a frozen system."""
    fixture, numbers, positions = load_reference(system, dtype)
    calc = GFN0Calculator(
        numbers,
        opts={"verbosity": 0, "exclude": exclude or []},
        dtype=dtype,
    )
    result = calc.singlepoint(
        positions,
        chrg=fixture["metadata"]["total_charge"],
    )
    return fixture, calc, result


def energy_partition(result: Result) -> dict[str, float]:
    """Collect the atomwise dxtb energy partition as summed scalars."""
    return {
        "eht": result.scf.sum().item(),
        "fermi": result.fenergy.sum().item(),
        **{
            reference: result.cenergies[label].sum().item()
            for reference, label in COMPONENTS.items()
        },
        "total": result.total.sum().item(),
    }


@pytest.mark.parametrize("system", SYSTEMS)
def test_float64_components_and_total(system: str) -> None:
    """Match every frozen component and total without cancellation masking."""
    fixture, calc, result = calculate(system, torch.float64)
    reference = fixture["energies"]
    actual = energy_partition(result)
    tolerances = fixture["metadata"]["tolerances"]

    assert result.iter == 0
    assert calc.interactions.labels == []
    assert result.potential.mono is not None
    assert torch.count_nonzero(result.potential.mono) == 0

    for component in COMPONENTS:
        assert actual[component] == pytest.approx(
            reference[component], abs=tolerances["component_atol"]
        )

    # Check the printed EHT/Fermi partition individually with its isolated
    # tolerance, then require their physically combined contribution tightly.
    assert actual["eht"] == pytest.approx(
        reference["eht"], abs=FLOAT64_ELECTRONIC_PARTITION_ATOL
    )
    assert actual["fermi"] == pytest.approx(
        reference["fermi"], abs=FLOAT64_ELECTRONIC_PARTITION_ATOL
    )
    assert actual["eht"] + actual["fermi"] == pytest.approx(
        reference["eht"] + reference["fermi"], abs=1.0e-9
    )
    assert actual["total"] == pytest.approx(
        reference["total"], abs=tolerances["total_atol"]
    )

    decomposed = sum(actual[key] for key in ("eht", "fermi", *COMPONENTS))
    assert actual["total"] == pytest.approx(decomposed, abs=1.0e-12)


@pytest.mark.parametrize("system", SYSTEMS)
def test_float32_observed_tolerance(system: str) -> None:
    """Preserve all components within the recorded float32 acceptance bound."""
    fixture, _, result = calculate(system, torch.float32)
    reference = fixture["energies"]
    actual = energy_partition(result)

    assert result.total.dtype == torch.float32
    assert result.scf.dtype == torch.float32
    for component, expected in reference.items():
        assert actual[component] == pytest.approx(expected, abs=FLOAT32_ATOL)


@pytest.mark.parametrize(
    ("system", "exclusion", "label"),
    [
        ("H2O", "ies", "IES"),
        ("H2O", "disp", "DispersionD4"),
        ("H2O", "rep", "Repulsion"),
        ("MB16_43_01", "srb", "ShortRangeBond"),
    ],
)
def test_public_component_exclusions(
    system: str, exclusion: str, label: str
) -> None:
    """Remove exactly the requested parameter-derived classical component."""
    _, _, baseline = calculate(system, torch.float64)
    _, excluded_calc, excluded = calculate(
        system, torch.float64, exclude=[exclusion]
    )

    removed = baseline.cenergies[label]
    assert label not in excluded_calc.classicals.labels
    assert label not in excluded.cenergies
    assert excluded.total.sum() == pytest.approx(
        baseline.total.sum() - removed.sum(), abs=1.0e-12
    )


def test_public_scf_exclusion() -> None:
    """Retain classical GFN0 terms when the electronic contribution is excluded."""
    _, _, baseline = calculate("H2O", torch.float64)
    _, _, excluded = calculate("H2O", torch.float64, exclude=["scf"])

    classical = torch.stack(list(baseline.cenergies.values())).sum(0)
    assert torch.count_nonzero(excluded.scf) == 0
    assert torch.count_nonzero(excluded.fenergy) == 0
    assert excluded.total == pytest.approx(classical, abs=1.0e-14)


def test_heterogeneous_packed_batch_matches_separate_calculations() -> None:
    """Match separate complete H2 and H2O calculations in a padded batch."""
    references = [
        load_reference(system, torch.float64) for system in ("H2", "H2O")
    ]
    numbers = pack([reference[1] for reference in references])
    positions = pack([reference[2] for reference in references])
    assert isinstance(numbers, torch.Tensor)
    assert isinstance(positions, torch.Tensor)

    charges = torch.tensor(
        [reference[0]["metadata"]["total_charge"] for reference in references],
        dtype=torch.float64,
    )
    batch_calc = GFN0Calculator(
        numbers, opts={"verbosity": 0}, dtype=torch.float64
    )
    batch = batch_calc.singlepoint(positions, chrg=charges)
    separate = [
        GFN0Calculator(
            reference[1], opts={"verbosity": 0}, dtype=torch.float64
        ).singlepoint(
            reference[2], chrg=reference[0]["metadata"]["total_charge"]
        )
        for reference in references
    ]

    assert batch.iter == 0
    assert batch.total.sum(-1) == pytest.approx(
        torch.stack([result.total.sum() for result in separate]), abs=1.0e-10
    )
    assert batch.scf.sum(-1) == pytest.approx(
        torch.stack([result.scf.sum() for result in separate]), abs=1.0e-10
    )
    for label in COMPONENTS.values():
        assert batch.cenergies[label].sum(-1) == pytest.approx(
            torch.stack([result.cenergies[label].sum() for result in separate]),
            abs=1.0e-10,
        )


def test_total_energy_autograd_smoke() -> None:
    """Keep a finite, nonzero position derivative through the full energy."""
    fixture, numbers, positions = load_reference("H2O", torch.float64)
    positions.requires_grad_(True)
    calc = GFN0Calculator(numbers, opts={"verbosity": 0}, dtype=torch.float64)

    energy = calc.energy(
        positions, chrg=fixture["metadata"]["total_charge"]
    ).sum()
    (gradient,) = torch.autograd.grad(energy, positions)

    assert energy.grad_fn is not None
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 1.0e-8
    assert gradient.sum(0) == pytest.approx(torch.zeros(3), abs=1.0e-12)
