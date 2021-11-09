import numpy as np
from ..utils import cellpar_to_cell

def density(entry):
    return entry.calculated_density

def chemical_name(entry):
    return entry.chemical_name

def chemical_name_as_html(entry):
    return entry.chemical_name_as_html

def refcode_family(entry):
    return entry.identifier[:6]

def formula(entry):
    return entry.formula

def is_organic(entry):
    return entry.is_organic

def polymorph(entry):
    return entry.polymorph

def cell_str_as_html(entry):
    a, b, c = entry.crystal.cell_lengths
    al, be, ga = entry.crystal.cell_angles
    return f'a {round(a, 2)}Å b {round(b, 2)}Å c {round(c, 2)}Å<br>α {round(al, 2)}° β {round(be, 2)}° γ {round(ga, 2)}°'

def crystal_system(entry):
    return entry.crystal.crystal_system

def spacegroup(entry):
    return entry.crystal.spacegroup_symbol

def molecular_centres(entry):
    frac_centres = []
    for c in entry.crystal.packing(inclusion='CentroidIncluded').components:
        coords = [a.fractional_coordinates for a in c.atoms]
        x, y, z = zip(*coords)
        frac_centres.append((sum(x)/len(coords), sum(y)/len(coords), sum(z)/len(coords)))
    frac_centres = np.array(frac_centres)
    frac_centres = np.mod(frac_centres, 1)
    site_diffs = np.abs(frac_centres[:, None] - frac_centres)
    overlapping = np.triu(np.all((site_diffs <= 1e-3) | (np.abs(site_diffs - 1) <= 1e-3), axis=-1), 1)
    if overlapping.any():
        keep_sites = ~overlapping.any(0)
        frac_centres = frac_centres[keep_sites]
        
    cell = cellpar_to_cell(*entry.crystal.cell_lengths, *entry.crystal.cell_angles)
    centres = frac_centres @ cell
    return centres