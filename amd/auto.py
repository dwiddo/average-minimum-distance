import warnings
import csv
import numpy as np
from .io import CifReader, CSDReader, SetWriter
from .calculate import PDD, AMD, PDD_to_AMD
from .utils import ETA

def analyse(cifs_or_refcodes,
            reader_kwargs=None,
            invariants=None,
            calc_kwargs=None,
            hdf5_path=None,
            csv_path=None):
    """
    Read cifs/CSD refcodes (auto-detected), optionally do any of the following:
    compute invariants, store periodic sets with invariants in .hdf5, write scalar
    data to a .csv.
    
    Parameters
    ----------
    cifs_or_refcodes : list or list of str
        Path to .cif(s) or refcode(s). Can be mixed.
    reader_kwargs : dict or None
        Keyword arguments to pass to the readers. (FIX-Must be shared between readers unless reader='ccdc'.)
    invariants : str or list of str or None
        'AMD' or 'PDD' or list with both. Calculates the chosen invariants. If None, calculate neither.
    calc_kwargs : dict or None
        Keyword arguments to pass to PDD (and AMD if 'k' is given). If None, k=100 is default. 
    hdf5_path : str or None
        If given, store the periodic sets to a .hdf5 file.
    csv_path : str or None
        If given, store all scalar data attached to the periodic sets to a .csv file.
    """
    
    if invariants is not None:
        
        if hdf5_path is None:
            warnings.warn('Invariants are computed but not used, so they will be skipped. '
                          'To calculate and store invariants, specify the .hdf5 path with parameter hdf5_path.')
            invariants = None
        
        if isinstance(invariants, str):
            invariants = [invariants]
        
        if calc_kwargs is not None and 'k' in calc_kwargs:
            k = calc_kwargs['k']
            del calc_kwargs['k']
        else:
            k = 100 # default
            
        if calc_kwargs is None:
            calc_kwargs = {}
    
    if isinstance(cifs_or_refcodes, str):
        cifs_or_refcodes = [cifs_or_refcodes]
    
    if reader_kwargs is None:
        cifreader_kwargs = csdreader_kwargs = {}
    else:
        cifreader_kwargs = reader_kwargs.copy()
        cifreader_kwargs.pop('families', None)
        csdreader_kwargs = reader_kwargs.copy()
        csdreader_kwargs.pop('reader', None)
    
    writer = None
    if hdf5_path is not None:
        writer = SetWriter(hdf5_path)
    
    data = []
    columns = []
    names = []
    
    from ccdc.io import EntryReader
    l = len(EntryReader('CSD'))
    
    for path_or_refcode in cifs_or_refcodes:
        
        if path_or_refcode.endswith('.cif'):
            reader = CifReader(path_or_refcode, **cifreader_kwargs)
        else:
            reader = CSDReader(path_or_refcode, **csdreader_kwargs)
        
        for i, periodic_set in enumerate(reader):
        
            # calc amds, pdds
            if invariants is not None:
                
                if 'PDD' in invariants:
                    pdd = PDD(periodic_set, k, **calc_kwargs)
                    periodic_set.tags['PDD'] = pdd
                    if 'AMD' in invariants:
                        periodic_set.tags['AMD'] = PDD_to_AMD(pdd)
                            
                elif 'AMD' in invariants:
                    periodic_set.tags['AMD'] = AMD(periodic_set, k)
            
            if csv_path is not None:
                row = {}
                for tag in periodic_set.tags:
                    value = periodic_set.tags[tag]
                    if np.isscalar(value):
                        if tag not in columns:
                            columns.append(tag)
                        row[tag] = value
                data.append(row)
                names.append(periodic_set.name)
            
            if writer is not None:
                writer.write(periodic_set)
            
            # verbose?
            # print(f'{i+1}/{l}', end='\r')
    
    if writer is not None:
        writer.close()
    
    if csv_path is not None:
        
        data_ = {}
        for col_name in columns:
            column_data = []
            for row in data:
                if col_name in row:
                    column_data.append(row[col_name])
                else:
                    column_data.append(None)
            data_[col_name] = column_data
            
        with open(csv_path, 'w', newline='', encoding='utf-8') as file:
            
            writer = csv.writer(file)
            
            writer.writerow(['', *columns])
            for name, row in zip(names, data):
                row_ = [name]
                for col_name in columns:
                    if col_name in row:
                        row_.append(row[col_name])
                    else:
                        row_.append('')
                writer.writerow(row_)
