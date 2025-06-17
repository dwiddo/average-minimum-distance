import json
from pathlib import Path

SUBS_DISORDER_TOL = 0.3
MAX_DISORDER_CONFIGS = 100

with open(str(Path(__file__).absolute().parent / "atomic_numbers.json")) as f:
    ATOMIC_NUMBERS = json.load(f)

with open(str(Path(__file__).absolute().parent / "atomic_masses.json")) as f:
    ATOMIC_MASSES = json.load(f)

ATOMIC_NUMBERS_TO_SYMS = {num: lab for lab, num in ATOMIC_NUMBERS.items()}
