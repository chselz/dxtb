"""Convert the reference GFN0 Fortran arrays into dxtb TOML parameters.

The converter intentionally reads only source arrays and reproduces the
mapping routines in ``gfn0_param.f90``.  It does not import or execute the
reference implementation and is not needed at dxtb runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from tad_mctc.units import AU2AA, AU2EV

MAX_ELEMENT = 86

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
    "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In "
    "Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf "
    "Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
).split()
ANGULAR_LABELS = "spdf"
NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
)


def without_comments(source: str) -> str:
    """Remove Fortran comments without changing array order."""
    return "\n".join(line.split("!", 1)[0] for line in source.splitlines())


def numbers(source: str) -> list[float]:
    """Parse Fortran numeric literals from an already isolated initializer."""
    source = re.sub(r"_[Ww][Pp]\b", "", source)
    return [
        float(value.replace("d", "e").replace("D", "E"))
        for value in NUMBER.findall(source)
    ]


def array(source: str, name: str) -> list[float]:
    """Extract a bracket array, including arrays wrapped in ``reshape``."""
    clean = without_comments(source)
    pattern = re.compile(
        rf"\b{re.escape(name)}\s*\([^=]*?\)\s*=\s*"
        rf"(?:aatoau\s*\*\s*)?(?:reshape\s*\(\s*)?\[",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(clean)
    if match is None:
        raise ValueError(f"Could not find Fortran array '{name}'.")
    end = clean.find("]", match.end())
    if end < 0:
        raise ValueError(f"Unterminated Fortran array '{name}'.")
    return numbers(clean[match.end() : end])


def data_array(source: str, name: str) -> list[float]:
    """Extract an old-style ``data name /.../`` initializer."""
    clean = without_comments(source)
    match = re.search(
        rf"\bdata\s+{re.escape(name)}\s*/(.*?)/",
        clean,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Could not find Fortran data array '{name}'.")
    return numbers(match.group(1))


def take(values: list[float], width: int, z: int) -> list[float]:
    """Take one Fortran column from a column-major rank-two array."""
    start = (z - 1) * width
    return values[start : start + width]


def primitive_count(z: int, angular: int, principal: int, valence: bool) -> int:
    """Reproduce ``setGFN0NumberOfPrimitives``."""
    if z <= 2:
        if angular == 0:
            return 3 if valence else 2
        if angular == 1:
            return 3
    elif angular == 0:
        return 6 if principal > 5 else 4
    elif angular == 1:
        return 6 if principal > 5 else 3
    elif angular in (2, 3):
        return 4
    raise ValueError(
        f"No GFN0 primitive-count rule for Z={z}, n={principal}, l={angular}."
    )


def d_block(z: int) -> bool:
    """Reproduce the reference ``dBlock`` selection exactly."""
    return 20 < z < 30 or 38 < z < 48 or 56 < z < 80


def parameter_data(param_source: str, srb_source: str) -> dict[str, object]:
    """Parse and transform all GFN0 parameter records."""
    raw = {
        name: array(param_source, name)
        for name in (
            "repAlpha",
            "repZeff",
            "eeqChi",
            "eeqGam",
            "eeqkCN",
            "eeqAlp",
            "nShell",
            "angShell",
            "principalQuantumNumber",
            "electronegativity",
            "shellPoly",
            "kCN",
            "selfEnergy",
            "slaterExponent",
            "kQShell",
            "kQAtom",
            "referenceOcc",
            "atomicRad",
        )
    }
    srb = {
        name: data_array(srb_source, name)
        for name in ("en", "r0", "cnfak", "p")
    }

    expected_sizes = {
        "repAlpha": 86,
        "repZeff": 86,
        "eeqChi": 86,
        "eeqGam": 86,
        "eeqkCN": 86,
        "eeqAlp": 86,
        "nShell": 86,
        "angShell": 3 * 86,
        "principalQuantumNumber": 3 * 86,
        "electronegativity": 86,
        "shellPoly": 4 * 86,
        "kCN": 3 * 86,
        "selfEnergy": 3 * 86,
        "slaterExponent": 3 * 86,
        "kQShell": 3 * 86,
        "kQAtom": 86,
        "referenceOcc": 3 * 86,
    }
    for name, expected in expected_sizes.items():
        if len(raw[name]) != expected:
            raise ValueError(
                f"{name}: expected {expected} values, found {len(raw[name])}."
            )
    if len(raw["atomicRad"]) != 118:
        raise ValueError("atomicRad must contain all 118 reference radii.")
    for name in ("en", "r0", "cnfak"):
        if len(srb[name]) != 86:
            raise ValueError(f"SRB {name} must contain 86 values.")
    if len(srb["p"]) != 8:
        raise ValueError("SRB period polynomial must contain eight values.")

    elements: dict[str, dict[str, object]] = {}
    for z, symbol in enumerate(SYMBOLS, 1):
        nshell = int(raw["nShell"][z - 1])
        angular = [int(value) for value in take(raw["angShell"], 3, z)[:nshell]]
        principal = [
            int(value)
            for value in take(raw["principalQuantumNumber"], 3, z)[:nshell]
        ]
        valence: list[bool] = []
        seen: set[int] = set()
        for ang in angular:
            valence.append(ang not in seen)
            seen.add(ang)

        shell_poly_angular = take(raw["shellPoly"], 4, z)
        kcn_angular = take(raw["kCN"], 3, z)
        kq_angular = take(raw["kQShell"], 3, z)
        refocc_angular = take(raw["referenceOcc"], 3, z)
        elements[symbol] = {
            "shells": [
                f"{n}{ANGULAR_LABELS[l]}" for n, l in zip(principal, angular)
            ],
            "levels": take(raw["selfEnergy"], 3, z)[:nshell],
            "slater": take(raw["slaterExponent"], 3, z)[:nshell],
            "ngauss": [
                primitive_count(z, l, n, is_valence)
                for l, n, is_valence in zip(angular, principal, valence)
            ],
            "refocc": [
                refocc_angular[l] if is_valence else 0.0
                for l, is_valence in zip(angular, valence)
            ],
            # dxtb's convention includes the 0.01 factor from shellPoly().
            "shpoly": [0.01 * shell_poly_angular[l] for l in angular],
            "kcn": [kcn_angular[l] for l in angular],
            "kq": [kq_angular[l] for l in angular],
            "kqat": raw["kQAtom"][z - 1],
            # The source literal is in Angstrom; store it in Bohr using the
            # canonical tad-mctc conversion.
            "h0rad": raw["atomicRad"][z - 1] / AU2AA,
            "zeff": raw["repZeff"][z - 1],
            "arep": raw["repAlpha"][z - 1],
            "en": raw["electronegativity"][z - 1],
            "eeq_chi": raw["eeqChi"][z - 1],
            "eeq_eta": raw["eeqGam"][z - 1],
            "eeq_kcn": raw["eeqkCN"][z - 1],
            "eeq_rad": raw["eeqAlp"][z - 1],
            "srb_r0": srb["r0"][z - 1],
            "srb_cnfak": srb["cnfak"][z - 1],
            "srb_en": srb["en"][z - 1],
        }

    angular_scale = [2.0, 2.4868, 2.27, 0.0]
    shell_scale = {
        ANGULAR_LABELS[i]
        + ANGULAR_LABELS[j]: 0.5 * (angular_scale[i] + angular_scale[j])
        for i in range(4)
        for j in range(i, 4)
    }
    pair_scale: dict[str, float] = {}
    coinage = {29, 47, 79}
    for iz in range(1, MAX_ELEMENT + 1):
        if not d_block(iz):
            continue
        for jz in range(1, iz + 1):
            if d_block(jz):
                pair_scale[f"{SYMBOLS[iz - 1]}-{SYMBOLS[jz - 1]}"] = (
                    0.9 if iz in coinage and jz in coinage else 1.1
                )

    return {
        "shell_scale": shell_scale,
        "pair_scale": pair_scale,
        "period1": srb["p"][:4],
        "period2": srb["p"][4:],
        "elements": elements,
    }


def scalar(value: object) -> str:
    """Format one TOML scalar deterministically."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.16E}"
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"Unsupported TOML value {value!r}.")


def vector(values: object) -> str:
    """Format one TOML array deterministically."""
    if not isinstance(values, list):
        raise TypeError("TOML vector input must be a list.")
    return "[" + ", ".join(scalar(value) for value in values) + "]"


def render(data: dict[str, object], param_hash: str, srb_hash: str) -> str:
    """Render the transformed data as a stable TOML document."""
    lines = [
        "# Generated by tools/convert_gfn0_parameters.py; do not edit manually.",
        "# SPDX-License-Identifier: LGPL-3.0-or-later",
        "# Source: sibling gfn0/src/gfn0_param.f90 and gfn0_srb.f90",
        f"# gfn0_param.f90 sha256: {param_hash}",
        f"# gfn0_srb.f90 sha256: {srb_hash}",
        f"# Unit conversions from tad-mctc: {AU2EV} eV/Eh, {AU2AA} Angstrom/Bohr",
        "",
        "[meta]",
        'name = "GFN0-xTB"',
        'reference = "P. Pracht, E. Caldeweyher, S. Ehlert, S. Grimme, ChemRxiv (2019), DOI: 10.26434/chemrxiv.8326202.v1"',
        "version = 0",
        "format = 1",
        "",
        "[hamiltonian.xtb]",
        "wexp = 1.0000000000000000E+00",
        "kdiff = 1.1241000000000001E+00",
        "enshell = [6.0000000000000000E-01, -1.0000000000000001E-01, -2.0000000000000001E-01, 0.0000000000000000E+00]",
        "enscale4 = 4.0000000000000000E+00",
        'cn = "erf"',
        "",
        "[hamiltonian.xtb.shell]",
    ]
    shell_scale = data["shell_scale"]
    assert isinstance(shell_scale, dict)
    lines.extend(
        f"{key} = {scalar(value)}" for key, value in shell_scale.items()
    )
    lines.extend(["", "[hamiltonian.xtb.kpair]"])
    pair_scale = data["pair_scale"]
    assert isinstance(pair_scale, dict)
    lines.extend(
        f"{key} = {scalar(value)}" for key, value in pair_scale.items()
    )
    lines.extend(
        [
            "",
            "[charge.eeq]",
            'cn = "erf"',
            "cutoff = 4.0000000000000000E+01",
            "cn_max = 8.0000000000000000E+00",
            "kcn = 7.5000000000000000E+00",
            "",
            "[repulsion.effective]",
            "kexp = 1.5000000000000000E+00",
            "klight = 1.5000000000000000E+00",
            "enscale = -8.9999999999999997E-02",
            "cutoff = 4.0000000000000000E+01",
            "",
            "[dispersion.d4]",
            "sc = false",
            "s6 = 1.0000000000000000E+00",
            "s8 = 2.8500000000000001E+00",
            "a1 = 8.0000000000000004E-01",
            "a2 = 4.5999999999999996E+00",
            "s9 = 0.0000000000000000E+00",
            "s10 = 0.0000000000000000E+00",
            "alp = 1.6000000000000000E+01",
            "",
            "[short_range.srb]",
            "shift = 5.3700000000000005E-02",
            "prefactor = -1.2900000000000000E-02",
            "steepness = 3.4847000000000001E+00",
            "enscale = 5.0969999999999993E-01",
            f"period1 = {vector(data['period1'])}",
            f"period2 = {vector(data['period2'])}",
            "cutoff2 = 2.0000000000000000E+02",
        ]
    )

    elements = data["elements"]
    assert isinstance(elements, dict)
    fields = (
        "shells",
        "levels",
        "slater",
        "ngauss",
        "refocc",
        "shpoly",
        "kcn",
        "kq",
        "kqat",
        "h0rad",
        "zeff",
        "arep",
        "en",
        "eeq_chi",
        "eeq_eta",
        "eeq_kcn",
        "eeq_rad",
        "srb_r0",
        "srb_cnfak",
        "srb_en",
    )
    for symbol in SYMBOLS:
        record = elements[symbol]
        assert isinstance(record, dict)
        lines.extend(["", f"[element.{symbol}]"])
        for field in fields:
            value = record[field]
            lines.append(
                f"{field} = {vector(value) if isinstance(value, list) else scalar(value)}"
            )

    return "\n".join(lines) + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    parser.add_argument("--gfn0-root", type=Path, default=workspace / "gfn0")
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "src/dxtb/_src/param/gfn0/gfn0-xtb.toml",
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if output is stale"
    )
    args = parser.parse_args()

    param_path = args.gfn0_root / "src/gfn0_param.f90"
    srb_path = args.gfn0_root / "src/gfn0_srb.f90"
    output = render(
        parameter_data(param_path.read_text(), srb_path.read_text()),
        sha256(param_path),
        sha256(srb_path),
    )
    if args.check:
        if not args.output.exists() or args.output.read_text() != output:
            raise SystemExit(
                f"Generated GFN0 parameter file is stale: {args.output}"
            )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
