# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
"""GFN0-xTB isotropic electrostatic classical contribution."""

from dxtb._src.components.classicals.ies import IES as IES
from dxtb._src.components.classicals.ies import new_ies as new_ies

__all__ = ["IES", "new_ies"]
