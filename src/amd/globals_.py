import json
from pathlib import Path

SUBS_DISORDER_TOL = 0.3
MAX_DISORDER_CONFIGS = 100

# Max distance between two sites considered to be refined at the same
# position, e.g. two elements sharing one crystallographic site.
EXACT_OVERLAP_TOL = 1e-3

# Tolerance on fractional occupancies, which are typically given to 2-4
# decimal places.
OCCUPANCY_TOL = 1e-6

# Max number of distinct occupancies to pair up by subset sum when
# inferring disorder groups. The search is exponential in this number.
MAX_OCCUPANCY_SUBSET_SEARCH = 20

# Tolerance used when testing whether an occupancy accounts for the site
# multiplicity of an atom on a special position, e.g. 0.5 for an atom on
# an inversion centre. Looser than OCCUPANCY_TOL since such occupancies
# are often rounded (0.33 for 1/3).
SPECIAL_POSITION_OCC_TOL = 1e-2

with open(str(Path(__file__).absolute().parent / "atomic_numbers.json")) as f:
    ATOMIC_NUMBERS: dict[str, int] = json.load(f)

with open(str(Path(__file__).absolute().parent / "atomic_masses.json")) as f:
    ATOMIC_MASSES: list[float] = json.load(f)

ATOMIC_NUMBERS_TO_SYMS = {num: lab for lab, num in ATOMIC_NUMBERS.items()}
