# GFN0-xTB parameter-data provenance

`gfn0-xtb.toml` is generated mechanically from `gfn0_param.f90` and
`gfn0_srb.f90` in the standalone GFN0 implementation at
<https://github.com/pprcht/gfn0>.

The source files state that the original parameter implementation is
Copyright (C) 2019–2020 Sebastian Ehlert and that the standalone adaptation is
Copyright (C) 2022–2023 Philipp Pracht. The source is distributed under the
GNU Lesser General Public License, version 3 or later. The generated TOML
retains that SPDX identifier and source hashes. The converter records the
legacy numerical constants and all transformations needed to reproduce the
artifact.

The parameter artifact is data-only: dxtb does not import, link, execute, or
otherwise depend on the sibling Fortran checkout at runtime or in its test
suite. The full LGPL text is available at
<https://www.gnu.org/licenses/lgpl-3.0.html> and in the source GFN0 checkout as
`COPYING.LESSER`.
