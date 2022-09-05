"""The command line interface for average-minimum-distance.
"""

import argparse
from .compare import compare


def main():

    parser = argparse.ArgumentParser(description='Compare crystals by PDD or AMD from the command line.')

    parser.add_argument('path_or_refcodes', type=str, nargs='+',
        help='path to a file or folder, or collection of refcodes (csd-python-api only).')
    parser.add_argument('--output', '-o', type=str, default='output', 
        help='name of the output file.')
    parser.add_argument('--format', '-f', type=str, default='csv', 
        help='format of the output file, default csv.')

    # args of amd.compare
    parser.add_argument('--by', '-b', type=str, default='AMD', choices=['AMD', 'PDD'],
        help='whether to use AMD or PDD to compare crystals.')
    parser.add_argument('--k', '-k', type=int, default=100, help='k value to use for AMD/PDD.')

    # reading args
    parser.add_argument('--reader', '-r', type=str, default='ase', 
        choices=['ase', 'ccdc', 'pycodcif'],
        help='backend package used for parsing files, default ase.')
    parser.add_argument('--remove_hydrogens', default=False, action='store_true',
        help='remove Hydrogen atoms from the crystals.')
    parser.add_argument('--disorder', type=str, default='skip', 
        choices=['skip', 'ordered_sites', 'all_sites'],
        help='control how disordered structures are handled.')
    parser.add_argument('--heaviest_component', default=False, action='store_true',
        help='remove all but the heaviest molecule in the asymmetric unit, intended for removing solvents (csd-python-api only).')
    parser.add_argument('--molecular_centres', default=False, action='store_true',
        help='uses the centres of molecules for comparisons instead of atoms (csd-python-api only).')
    parser.add_argument('--supress_warnings', default=False, action='store_true',
        help='do not show warnings encountered during reading.')
    parser.add_argument('--families', default=False, action='store_true',
        help='interpret strings given as refcode families (csd-python-api only).')

    # pdd args
    parser.add_argument('--collapse_tol', type=float, default=1e-4,
        help='tolerance for collapsing rows of PDDs.')

    # compare args
    parser.add_argument('--metric', type=str, default='chebyshev',
        help='metric used to compare AMDs/rows of PDDs.')
    parser.add_argument('--n_jobs', type=int, default=1,
        help='number of cores to use for multiprocessing.')
    parser.add_argument('--verbose', type=int, default=0,
        help='tolerance for collapsing rows of PDDs.')
    parser.add_argument('--low_memory', default=False, action='store_true',
        help='use an alternative algorithm with lower memory usage.')

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
    elif len(path_or_refcodes) == 2:
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