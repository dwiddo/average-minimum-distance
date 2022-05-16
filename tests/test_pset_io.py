import os
import pytest
import amd


def test_SetReader(root_dir, reference_data):
    structs = reference_data['CSD_families']
    
    for dataset, structs in reference_data.items():
        path = os.path.join(root_dir, rf'{dataset}.hdf5')
        for s, s_ in zip(amd.SetReader(path), structs):
            if not s == s_:
                pytest.fail(f'On structure {s.name} SetReader disagrees with reference')


def test_SetWriter(root_dir, reference_data):

    for name in reference_data:
        refs = reference_data[name]

        path = os.path.join(root_dir, rf'{name}_TEMP.hdf5')

        with amd.SetWriter(path) as writer:
            writer.iwrite(refs)

        with amd.SetReader(path) as reader_:
            for s, s_ in zip(refs, reader_):
                if not s == s_:
                    pytest.fail(f'Structure {s_.name} written with SetWriter disagrees with reference')

        os.remove(path)
