"""Command line interface for :func:`amd.compare() <.compare.compare>`.
"""

import argparse
import sys
from .compare import compare


def main():
    """Entry point for command line interface for
    :func:`amd.compare() <.compare.compare>.
    """

    parser = argparse.ArgumentParser(
        description='Compare crystals by PDD or AMD from the command line.'
    )

    parser.add_argument('path_or_refcodes', type=str, nargs='+',
        help='(str) One or two paths to files or folders, or a collection ' \
             'of CSD refcodes if csd-python-api is installed.')
    parser.add_argument('--output', '-o', type=str, default='output',
        help='(str) Path of the output file.')
    parser.add_argument('--format', '-f', type=str, default='csv', 
        help='(str) Format of the output file, default csv.')

    # args of amd.compare
    parser.add_argument(
        '--by', '-b', type=str, default='AMD', choices=['AMD', 'PDD'],
        help='(str) Use AMD or PDD to compare crystals.')
    parser.add_argument('--k', '-k', type=int, default=100,
        help='(int) Number of neighbour atoms to use for AMD/PDD.')
    parser.add_argument('--nearest', '-n', type=int, default=None,
        help='(int) Find n nearest neighbours instead of a full distance ' \
             'matrix between crystals.')

    # reading args
    parser.add_argument('--reader', '-r', type=str, default='ase',
        choices=['ase', 'pycodcif', 'ccdc', 'pymatgen', 'gemmi'],
        help='(str) backend package used for parsing files, default ase.')
    parser.add_argument('--remove_hydrogens', default=False,
        action='store_true', help='(flag) remove Hydrogen atoms.')
    parser.add_argument('--disorder', type=str, default='skip',
        choices=['skip', 'ordered_sites', 'all_sites'],
        help='(str) control how disordered structures are handled.')
    parser.add_argument(
        '--heaviest_component', default=False, action='store_true',
        help='(flag) (csd-python-api only) keep only the heaviest part of ' \
             'the asymmetric unit, intended for removing solvents.')
    parser.add_argument(
        '--molecular_centres', default=False, action='store_true',
        help='(flag) (csd-python-api only) uses the centres of molecules ' \
             'for comparisons instead of atoms.')
    parser.add_argument(
        '--supress_warnings', default=False, action='store_true',
        help='(flag) do not show warnings encountered during reading.')
    parser.add_argument('--families', default=False, action='store_true',
        help='(flag) (csd-python-api only) interpret path_or_refcodes as ' \
             'refcode families.')

    # PDD args
    parser.add_argument('--collapse_tol', type=float, default=1e-4,
        help='(float) tolerance for collapsing rows of PDDs.')

    # compare args
    parser.add_argument('--metric', type=str, default='chebyshev',
        help='(str) metric used to compare AMDs/rows of PDDs, default chebyshev.')
    parser.add_argument('--n_jobs', type=int, default=1,
        help='(int) number of cores to use for multiprocessing in PDD comaprisons.')
    parser.add_argument('--verbose', type=int, default=0,
        help='(int) print an ETA to the terminal for PDD comparisons. Passed to joblib.Parallel if using multiprocessing.')
    parser.add_argument('--low_memory', default=False, action='store_true',
        help='(flag) use an alternative slower algorithm for AMD comparisons which uses less memory.')

    # remove some arguments before passing others to amd.compare
    kwargs = vars(parser.parse_args())
    path_or_refcodes = kwargs.pop('path_or_refcodes')
    outpath = kwargs.pop('output', 'output')
    ext = kwargs.pop('format', 'csv')

    # parameter is show_warnings, here it's a flag supress_warnings
    kwargs['show_warnings'] = not kwargs['supress_warnings']
    kwargs.pop('supress_warnings', None)

    crystals = path_or_refcodes[0]
    crystals_ = None
    if len(path_or_refcodes) == 2:
        crystals_ = path_or_refcodes[1]
    elif len(path_or_refcodes) > 2:
        msg = 'amd.compare accepts one or two collections of crystals for comparison.'
        raise ValueError(msg)

    df = compare(crystals, crystals_, **kwargs)

    if kwargs['verbose']:
        sys.stdout.write(df)

    if not outpath.endswith('.' + ext):
        outpath += '.' + ext

    try:
        output_func = getattr(df, 'to_' + ext)
        output_func(outpath)
    except AttributeError:
        sys.stdout.write(f'Unknown format {ext}, using csv.')
        df.to_csv(outpath + '.csv')


if __name__ == '__main__':
    main()
