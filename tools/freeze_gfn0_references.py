"""Freeze compact, self-contained GFN0 numerical oracles from xtb outputs.

This is a maintainer tool. Tests consume only the generated JSON files and do
not access the sibling ``references`` directory or a Fortran executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SYMBOL_TO_NUMBER = {
    symbol: z
    for z, symbol in enumerate(
        (
            "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr "
            "Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd "
            "Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er "
            "Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
        ).split()
    )
}
ENERGY_LABELS = {
    "total energy": "total",
    "H0 energy": "eht",
    "repulsion energy": "repulsion",
    "electrostat energy": "ies",
    "dispersion energy": "d4_2body",
    "short-range corr.": "srb",
}


def coordinates(path: Path) -> tuple[list[int], list[str], list[list[float]]]:
    nums: list[int] = []
    syms: list[str] = []
    xyz: list[list[float]] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 4 or fields[0].startswith("$"):
            continue
        symbol = fields[3].capitalize()
        nums.append(SYMBOL_TO_NUMBER[symbol])
        syms.append(symbol)
        xyz.append([float(value.rstrip(",")) for value in fields[:3]])
    if not nums:
        raise ValueError(f"No coordinates found in {path}.")
    return nums, syms, xyz


def energies(output: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for label, key in ENERGY_LABELS.items():
        matches = re.findall(
            rf"::\s*{re.escape(label)}\s+([-+]?\d+\.\d+)\s+Eh", output
        )
        if not matches:
            raise ValueError(f"Missing '{label}' in xtb output.")
        result[key] = float(matches[-1])
    # The displayed H0 energy already includes the Fermi free-energy term for
    # these nondegenerate 300 K references; it is numerically zero separately.
    result["fermi"] = 0.0
    return result


def orbitals(output: str) -> list[dict[str, float | int]]:
    header = output.find("#    Occupation            Energy/Eh")
    if header < 0:
        raise ValueError("Missing orbital table in xtb output.")
    end = output.find("HL-Gap", header)
    table = output[header:end]
    occupied = re.compile(
        r"^\s*(\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+",
        re.MULTILINE,
    )
    unoccupied = re.compile(
        r"^\s*(\d+)\s{5,}([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+",
        re.MULTILINE,
    )
    values: dict[int, dict[str, float | int]] = {}
    for match in occupied.finditer(table):
        index = int(match.group(1))
        values[index] = {
            "index": index,
            "occupation": float(match.group(2)),
            "energy": float(match.group(3)),
        }
    for match in unoccupied.finditer(table):
        index = int(match.group(1))
        if index not in values:
            values[index] = {
                "index": index,
                "occupation": 0.0,
                "energy": float(match.group(2)),
            }
    return [values[index] for index in sorted(values)]


def metadata(output: str) -> dict[str, object]:
    version = re.search(r"xtb version\s+(\S+)\s+\(([^)]+)\)", output)
    command = re.search(r"program call\s*:\s*(.+)", output)
    if version is None or command is None:
        raise ValueError("Could not extract xtb provenance metadata.")
    command_text = command.group(1).strip()
    charge_match = re.search(r"--chrg\s+([-+]?\d+)", command_text)
    total_charge = int(charge_match.group(1)) if charge_match else 0
    return {
        "oracle": "xtb",
        "oracle_version": version.group(1),
        "oracle_revision": version.group(2),
        "command": command_text,
        "total_charge": total_charge,
        "spin": 0,
        "coordinate_unit": "bohr",
        "energy_unit": "hartree",
        "orbital_energy_unit": "hartree",
        "charge_unit": "elementary_charge",
        "legacy_conversion_constants_used": True,
        "ev_per_hartree": 27.21138505,
        "angstrom_per_bohr": 0.52917726,
        "tolerances": {
            "component_atol": 1.0e-8,
            "total_atol": 1.0e-9,
            "charge_atol": 5.0e-8,
            "orbital_energy_atol": 5.0e-8,
        },
    }


def freeze(system: Path) -> dict[str, object]:
    output = (system / "xtb.out").read_text()
    nums, syms, xyz = coordinates(system / "coord")
    charges = [
        float(value) for value in (system / "charges").read_text().split()
    ]
    if len(charges) != len(nums):
        raise ValueError(
            f"Charge count does not match atom count for {system.name}."
        )
    return {
        "schema_version": 1,
        "system": system.name,
        "metadata": metadata(output),
        "numbers": nums,
        "symbols": syms,
        "positions": xyz,
        "atomic_charges": charges,
        "energies": energies(output),
        "orbitals": orbitals(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--references", type=Path, default=project.parent / "references"
    )
    parser.add_argument(
        "--output", type=Path, default=project / "test/test_gfn0/refs"
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if fixtures are stale"
    )
    args = parser.parse_args()

    generated: dict[str, str] = {}
    for system in sorted(
        path for path in args.references.iterdir() if path.is_dir()
    ):
        text = json.dumps(freeze(system), indent=2, sort_keys=True) + "\n"
        filename = f"{system.name}.json"
        generated[filename] = text

    manifest = {
        "schema_version": 1,
        "fixtures": {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in generated.items()
        },
    }
    generated["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    if args.check:
        stale = [
            name
            for name, text in generated.items()
            if not (args.output / name).exists()
            or (args.output / name).read_text() != text
        ]
        if stale:
            raise SystemExit(
                f"Stale GFN0 reference fixtures: {', '.join(stale)}"
            )
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    for name, text in generated.items():
        (args.output / name).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
