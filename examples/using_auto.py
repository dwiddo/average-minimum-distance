"""
Demonstration of amd.auto.analyse() which given cifs and/or CSD refcodes
reads the structures, and optionally extracts data, calculates AMDs/PDDs,
writes the structures to a file and makes a .csv of the scalar data.
"""

import os
import amd
import amd.auto
import amd.ccdc_utils as cu
import warnings
warnings.filterwarnings('ignore')

# cifs or refcodes here (or 'CSD' for the whole CSD)
cifs_or_refcodes = [
    'CSD'
]

folder = ''
hdf5_file = 'crystals.hdf5'
csv_file  = 'crystal_data.csv'

extract_data = {
    'refcode_family':           cu.refcode_family,
    'chemical_name':            cu.chemical_name,
    'chemical_name_as_html':    cu.chemical_name_as_html,
    'density':                  cu.density,
    'formula':                  cu.formula,
    'cell_str':                 cu.cell_str_as_html,
    'crystal_system':           cu.crystal_system,
    'spacegroup':               cu.spacegroup,
    'is_organic':               cu.is_organic,
    'polymorph':                cu.polymorph,
    # 'molecular_centres':        cu.molecular_centres,
}

include_if = [
    # cu.has_disorder
]

reader_kwargs = {
    'reader':               'ccdc',
    'remove_hydrogens':     False,      
    'disorder':             'all_sites',
    'heaviest_component':   False,
    'extract_data':         extract_data,
    'include_if':           include_if,
    'families':             False,
}

invariants = [
    'AMD',
    # 'PDD',
]

# defaults
calc_kwargs = {
    'k':            100,
    'order':        1,
    'lexsort':      True,
    'collapse':     True, 
    'collapse_tol': 1e-4,
}

hdf5_path = os.path.join(folder, hdf5_file)
csv_path  = os.path.join(folder, csv_file)

amd.auto.analyse(cifs_or_refcodes,
                 reader_kwargs=reader_kwargs,
                 invariants=invariants,
                 calc_kwargs=calc_kwargs,
                 hdf5_path=hdf5_path,
                 csv_path=csv_path)

# structures = list(amd.SetReader(hdf5_path))