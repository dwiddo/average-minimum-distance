"""Stores the tag strings used in CIFs.
"""

CIF_TAGS = {

    'cellpar': [
        '_cell_length_a', 
        '_cell_length_b', 
        '_cell_length_c',
        '_cell_angle_alpha', 
        '_cell_angle_beta', 
        '_cell_angle_gamma',
    ],

    'atom_site_fract': [
        '_atom_site_fract_x',
        '_atom_site_fract_y',
        '_atom_site_fract_z',
    ],

    'atom_site_cartn': [
        '_atom_site_cartn_x',
        '_atom_site_cartn_y',
        '_atom_site_cartn_z',
    ],

    'symop': [
        '_symmetry_equiv_pos_as_xyz',
        '_space_group_symop_operation_xyz',
        '_space_group_symop.operation_xyz',
        '_symmetry_equiv_pos_as_xyz_',
        '_space_group_symop_operation_xyz_',
    ],

    'atom_symbol': [
        '_atom_site_type_symbol',
        '_atom_site_label',
    ],

    'spacegroup_name' : [
        '_symmetry_space_group_name_H-M',
        '_symmetry_space_group_name_H_M',
        '_symmetry_space_group_name_H-M_',
        '_symmetry_space_group_name_H_M_',
        '_space_group_name_Hall',
        '_space_group_name_Hall_',
        '_space_group_name_H-M_alt',
        '_space_group_name_H-M_alt_',
        '_symmetry_space_group_name_hall',
        '_symmetry_space_group_name_hall_',
        '_symmetry_space_group_name_h-m',
        '_symmetry_space_group_name_h-m_',
    ],

    'spacegroup_number': [
        '_space_group_IT_number',
        '_symmetry_Int_Tables_number',
        '_space_group_IT_number_',
        '_symmetry_Int_Tables_number_',
    ],

}
