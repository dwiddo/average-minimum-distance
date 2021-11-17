import csv
import numpy as np
from .io import CifReader, CSDReader, SetWriter
from .calculate import PDD, AMD, PDD_to_AMD

def analyse(cifs_or_refcodes,
            reader_kwargs=None,
            invariants=None,
            calc_kwargs=None,
            hdf5_path=None,
            csv_path=None):
    
    if invariants is not None:
        if calc_kwargs is not None:
            k = calc_kwargs['k']
            del calc_kwargs['k']
        else:
            k = 100 # default
    
    if isinstance(cifs_or_refcodes, str):
        cifs_or_refcodes = [cifs_or_refcodes]
    
    writer = None
    if hdf5_path is not None:
        writer = SetWriter(hdf5_path)
    
    data = []
    columns = []
    names = []
    
    for path_or_refcode in cifs_or_refcodes:
        
        if path_or_refcode.endswith('.cif'):
            reader = CifReader(path_or_refcode, **reader_kwargs)
        else:
            reader = CSDReader(path_or_refcode, **reader_kwargs)

        for periodic_set in reader:
        
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

def analyse_hdf5():
    pass

