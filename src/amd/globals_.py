import json
from pathlib import Path

SUBS_DISORDER_TOL = 0.3
MAX_DISORDER_CONFIGS = 100

# Max distance between two sites considered to be refined at the same
# position, e.g. two elements sharing one crystallographic site.
EXACT_OVERLAP_TOL = 0.001

# Tolerance on fractional occupancies
OCCUPANCY_TOL = 1e-6

# Slack allowed when testing whether occupancies sum to more than one,
# deposited values can be rounded (0.875/0.125 arrives as 0.88/0.13, LEYQUD)
OCCUPANCY_ROUNDING_TOL = 0.02

# Max number of distinct occupancies to pair up by subset sum when
# inferring disorder groups. The search is exponential in this number.
MAX_OCCUPANCY_SUBSET_SEARCH = 20

# Occupancy ≈ multiplicity / n_syms means a special position, not disorder
# (0.33 given for 1/3).
SPECIAL_POSITION_OCC_TOL = 0.01

# Linkage distance under which disordered sites belong to one region of
# disorder, used to split sites into assemblies when occupancies cannot
# (a suppressed model with a 0.5:0.5 split gives every site the same occupancy).
DISORDER_REGION_TOL = 2.0

# How far a disorder alternative may sit from the nearest site kept by
# heaviest_component and still be taken as an alternative of it. ccdc's
# disorder model describes the whole asymmetric unit, so it must be told
# apart from an alternative of a solvent which was thrown away. Generous
# next to any bond length, since the two describe one region of the
# crystal.
DISORDER_COMPONENT_TOL = 3.0

# dict mapping atomic symbols to numbers {"H": 1, ...}
with open(str(Path(__file__).absolute().parent / "atomic_numbers.json")) as f:
    ATOMIC_NUMBERS: dict[str, int] = json.load(f)

# list of atomic masses indexed by atomic number
with open(str(Path(__file__).absolute().parent / "atomic_masses.json")) as f:
    ATOMIC_MASSES: list[float] = json.load(f)

ATOMIC_NUMBERS_TO_SYMS = {num: lab for lab, num in ATOMIC_NUMBERS.items()}
