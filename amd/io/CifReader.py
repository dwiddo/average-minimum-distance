from ._Reader import _Reader
from ase.io.cif import parse_cif                # .cif -> Iterable[CIFBlock]

try:
    from ccdc.io import EntryReader             # .cif or 'CSD' -> Iterable[Entry]
    _CCDC_ENABLED = True
except ImportError:
    _CCDC_ENABLED = False

import warnings
def _warning(message, category, filename, lineno, file=None, line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'
warnings.formatwarning = _warning

class CifReader(_Reader):
    """Read all structures in a .cif with ase or ccdc, yielding PeriodicSet objects."""
    
    READERS = {'ase', 'ccdc'}
    
    def __init__(self, filename, reader='ase', **kwargs):
        
        super().__init__(**kwargs)
        
        if reader not in CifReader.READERS:
            raise ValueError(f'Invalid reader {reader}. Reader must be one of {CifReader.READERS}')

        if self.heaviest_component and reader != 'ccdc':
            raise NotImplementedError(f'Parameter heaviest_component not implimented for {reader}, only ccdc.')
        
        # function _map from parent _Reader sets up self._generator which yields PeriodicSet objects

        if reader == 'ase':
            
            self._generator = self._map(self._CIFBlock_to_PeriodicSet, parse_cif(filename))
            
        elif reader == 'ccdc':
            
            if not _CCDC_ENABLED:
                raise ImportError(f"Failed to import ccdc. Either it is not installed or not licensed.")
            
            self._generator = self._map(self._Entry_to_PeriodicSet, EntryReader(filename))
