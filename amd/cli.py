"""Command line interface for :func:`amd.compare() <.compare.compare>`.
"""

import argparse
from .compare import compare


def main():
    """Entry point for command line interface for :func:`amd.compare() <.compare.compare>."""

    parser = argparse.ArgumentParser(description='Compare crystals by PDD or AMD from the command line.')

    parser.add_argument('path_or_refcodes', type=str, nargs='+',
        help='(str) Path to a file or folder, or a collection of refcodes (csd-python-api only).')
    parser.add_argument('--output', '-o', type=str, default='output',
        help='(str) Path of the output file.')
    parser.add_argument('--format', '-f', type=str, default='csv', 
        help='(str) Format of the output file, default csv.')

    # args of amd.compare
    parser.add_argument('--by', '-b', type=str, default='AMD', choices=['AMD', 'PDD'],
        help='(str) Whether to use AMD or PDD to compare crystals.')
    parser.add_argument('--k', '-k', type=int, default=100,
        help='(int) Value of k (number of nearest neighbours) to use for AMD/PDD.')
    parser.add_argument('--nearest', '-n', type=int, default=None,
        help='(int) Find n nearest neighbours instead of a full distance matrix.')

    # reading args
    parser.add_argument('--reader', '-r', type=str, default='ase',
        choices=['ase', 'ccdc', 'pycodcif'],
        help='(str) backend package used for parsing files, default ase.')
    parser.add_argument('--remove_hydrogens', default=False, action='store_true',
        help='(flag) remove Hydrogen atoms from the crystals.')
    parser.add_argument('--disorder', type=str, default='skip',
        choices=['skip', 'ordered_sites', 'all_sites'],
        help='(str) control how disordered structures are handled.')
    parser.add_argument('--heaviest_component', default=False, action='store_true',
        help='(flag) remove all but the heaviest molecule in the asymmetric unit, intended for removing solvents (csd-python-api only).')
    parser.add_argument('--molecular_centres', default=False, action='store_true',
        help='(flag) uses the centres of molecules for comparisons instead of atoms (csd-python-api only).')
    parser.add_argument('--supress_warnings', default=False, action='store_true',
        help='(flag) do not show warnings encountered during reading.')
    parser.add_argument('--families', default=False, action='store_true',
        help='(flag) interpret strings given as refcode families (csd-python-api only).')

    # PDD args
    parser.add_argument('--collapse_tol', type=float, default=1e-4,
        help='(float) tolerance for collapsing rows of PDDs.')

    # compare args
    parser.add_argument('--metric', type=str, default='chebyshev',
        help='(str) metric used to compare AMDs/rows of PDDs.')
    parser.add_argument('--n_jobs', type=int, default=1,
        help='(int) number of cores to use for multiprocessing.')
    parser.add_argument('--verbose', type=int, default=0,
        help='(int) tolerance for collapsing rows of PDDs.')
    parser.add_argument('--low_memory', default=False, action='store_true',
        help='(flag) use an alternative slower algorithm with a lower memory footprint.')

    args = parser.parse_args()
    kwargs = vars(args)
    path_or_refcodes = kwargs.pop('path_or_refcodes')
    outpath = kwargs.pop('output', 'output')
    ext = kwargs.pop('format', 'csv')
    kwargs['show_warnings'] = not kwargs['supress_warnings']
    kwargs.pop('supress_warnings', None)

    crystals = path_or_refcodes[0]
    crystals_ = None
    if len(path_or_refcodes) > 2:
        raise ValueError('amd.compare: one or two collections of crystals are allowed for comparison.')
    if len(path_or_refcodes) == 2:
        crystals_ = path_or_refcodes[1]

    df = compare(crystals, crystals_, **kwargs)

    if kwargs['verbose']:
        print(df)

    if not outpath.endswith('.' + ext):
        outpath += '.' + ext

    try:
        output_func = getattr(df, 'to_' + ext)
        output_func(outpath)
    except AttributeError:
        print(f'Cannot output format {ext}, using csv instead.')
        df.to_csv(outpath + '.csv')


if __name__ == '__main__':
    main()
