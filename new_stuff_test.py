import amd
from amd.io import ccdc_utils
import pandas as pd 

extract_data = {
    'Density': ccdc_utils.density,
    'Chemical name': ccdc_utils.chemical_name_as_html,
    'Family': ccdc_utils.refcode_family,
    'Formula': ccdc_utils.formula,
    'Is organic': ccdc_utils.is_organic,
    'Polymorph': ccdc_utils.polymorph,
    'Cell': ccdc_utils.cell_str_as_html,
    'Crystal system': ccdc_utils.crystal_system,
    'Spacegroup': ccdc_utils.spacegroup,
    'Molecular centres': ccdc_utils.molecular_centres,
}

reader = amd.CSDReader(['DEBXIT', 'GLYCIN'], families=True, extract_data=extract_data)

with amd.SetWriter('DEBXIT_test.hdf5') as writer:
    writer.iwrite(reader)

with amd.SetReader('DEBXIT_test.hdf5') as reader:
    data = reader.to_dict()
    df = pd.DataFrame(data, index=reader.keys())

df.to_csv('DEBXIT_test_csv.csv')
