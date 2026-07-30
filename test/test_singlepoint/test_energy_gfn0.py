# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""Public API and end-to-end routing tests for GFN0-xTB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from tad_mctc import read

import dxtb
from dxtb import GFN0_XTB, Calculator
from dxtb._src.calculators.config import Config
from dxtb._src.calculators.result import Result
from dxtb._src.cli import Driver, parser
from dxtb._src.components.interactions import Interaction
from dxtb._src.constants import defaults, labels
from dxtb._src.xtb.gfn0 import GFN0Hamiltonian
from dxtb.calculators import GFN0Calculator

REFS = Path(__file__).parents[1] / "test_gfn0" / "refs"
CLI_COORD = Path(__file__).parent / "mols" / "H2O" / "coord"


def _h2o() -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    fixture = json.loads((REFS / "H2O.json").read_text())
    numbers = torch.tensor(fixture["numbers"])
    positions = torch.tensor(fixture["positions"], dtype=torch.float64)
    return fixture, numbers, positions


def test_public_exports() -> None:
    """Expose the built-in parameter and thin calculator publicly."""
    assert dxtb.GFN0_XTB is GFN0_XTB
    assert dxtb.calculators.GFN0Calculator is GFN0Calculator
    assert "GFN0_XTB" in dxtb.__all__
    assert "GFN0Calculator" in dxtb.calculators.__all__


@pytest.mark.parametrize("method", [labels.GFN0_XTB, *labels.GFN0_XTB_STRS])
def test_config_accepts_gfn0(method: int | str) -> None:
    """Accept integer zero and every documented GFN0 alias."""
    config = Config(method=method)
    assert config.method == labels.GFN0_XTB
    assert config.scf.method == labels.GFN0_XTB
    assert config.scf.requires_iterations is False


def test_cli_method_choices() -> None:
    """Make every GFN0 alias available through argparse."""
    assert set(labels.GFN0_XTB_STRS) <= set(defaults.METHOD_CHOICES)
    for method in labels.GFN0_XTB_STRS:
        args = parser().parse_args(["--method", method, str(CLI_COORD)])
        assert args.method == method


def test_thin_and_generic_calculator_match_reference() -> None:
    """Route both public calculator forms through the same GFN0 model."""
    fixture, numbers, positions = _h2o()
    opts = {
        "verbosity": 0,
        "cache_enabled": True,
        "cache_iterations": True,
    }

    thin = GFN0Calculator(numbers, opts=opts, dtype=torch.float64)
    generic = Calculator(numbers, GFN0_XTB, opts=opts, dtype=torch.float64)

    for calc in (thin, generic):
        assert calc.opts.method == labels.GFN0_XTB
        assert calc.opts.ints.level == labels.INTLEVEL_HCORE
        assert isinstance(calc.integrals.hcore, GFN0Hamiltonian)
        assert calc.interactions.labels == []
        assert set(calc.classicals.labels) == {
            "DispersionD4",
            "IES",
            "Repulsion",
            "ShortRangeBond",
        }

    thin_result = thin.singlepoint(positions)
    generic_result = generic.singlepoint(positions)
    reference = fixture["energies"]["total"]

    assert thin_result.total.sum().item() == pytest.approx(
        reference, abs=1.0e-9
    )
    assert torch.equal(thin_result.total, generic_result.total)
    assert torch.equal(thin_result.scf, generic_result.scf)
    assert thin_result.iter == generic_result.iter == 0
    assert thin.cache.iterations is not None
    assert generic.cache.iterations is not None
    assert int(thin.cache.iterations) == int(generic.cache.iterations) == 0


def test_gfn0_automatic_integral_level() -> None:
    """Raise an explicitly low integral level only to HCORE for GFN0."""
    _, numbers, _ = _h2o()
    calc = GFN0Calculator(
        numbers,
        opts={"verbosity": 0, "int_level": labels.INTLEVEL_NONE},
        dtype=torch.float64,
    )
    assert calc.opts.ints.level == labels.INTLEVEL_HCORE
    assert calc.integrals.dipole is None
    assert calc.integrals.quadrupole is None


def test_public_interaction_rejection() -> None:
    """Reject custom density-dependent interactions through the public API."""
    _, numbers, positions = _h2o()
    calc = GFN0Calculator(
        numbers,
        interaction=Interaction(),
        opts={"verbosity": 0},
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="does not support charge-dependent"):
        calc.singlepoint(positions)


@pytest.mark.parametrize(
    "entrypoint", ["forces_analytical", "dipole_analytical"]
)
def test_analytical_entrypoints_fail_fast(entrypoint: str) -> None:
    """Reject incomplete analytical GFN0 properties before prerequisites."""
    _, numbers, positions = _h2o()
    calc = GFN0Calculator(numbers, opts={"verbosity": 0}, dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="GFN0-xTB"):
        getattr(calc, entrypoint)(positions)


def test_cli_gfn0_matches_generic_calculator() -> None:
    """Select the built-in GFN0 parameter through the real CLI driver."""
    args = parser().parse_args(
        [
            "--method",
            "gfn0",
            "--verbosity",
            "0",
            "--dtype",
            "float64",
            str(CLI_COORD),
        ]
    )
    cli_result = Driver(args).singlepoint()
    assert isinstance(cli_result, Result)

    numbers, positions = read(CLI_COORD, dtype=torch.float64)
    calc = Calculator(
        numbers,
        GFN0_XTB,
        opts={"verbosity": 0},
        dtype=torch.float64,
    )
    generic_result = calc.singlepoint(positions)

    assert torch.equal(cli_result.total, generic_result.total)
    assert cli_result.iter == generic_result.iter == 0
