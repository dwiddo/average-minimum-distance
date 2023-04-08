import pytest
import amd
import numpy as np


def test_diameter():

    hex_cell = np.identity(3)
    hex_cell[1, 0] = -0.5
    hex_cell[1, 1] = np.sin(np.deg2rad(120.0))
    cells_and_diams = [(np.identity(3), np.sqrt(3)), (hex_cell, 2)]
    for cell, diam in cells_and_diams:
        d = amd.diameter(cell)
        if not np.abs(d - diam) < 1e-15:
            pytest.fail(f'amd.diameter() of {cell} should be {diam}, got {d}')


def test_cellpar_to_cell():

    hex_cellpar = np.array([1.0, 1.0, 1.0, 90.0, 90.0, 120.0])
    hex_cell = np.identity(3)
    hex_cell[1, 0] = -0.5
    hex_cell[1, 1] = np.sin(np.deg2rad(120.0))
    
    cellpar_2 = np.array([1.0, 2.0, 3.0, 60.0, 90.0, 120.0])
    cell_2 = np.identity(3)
    cell_2[1, 0] = -1
    cell_2[1, 1] = 2 * np.sin(np.deg2rad(cellpar_2[5]))
    cy = 0.5 / np.sin(np.deg2rad(120))
    cell_2[2, 1] = 3 * cy
    cell_2[2, 2] = 3 * np.sqrt(1.0 - cy ** 2)
    cellpars_and_cells = [(hex_cellpar, hex_cell), (cellpar_2, cell_2)]

    for cellpar, cell in cellpars_and_cells:
        cell_ = amd.cellpar_to_cell(cellpar)
        if not np.allclose(cell, cell_):
            pytest.fail(
            f'amd.cellpar_to_cell() of {cellpar} should be {hex_cell}, got '
            f'{cell}'
        )
