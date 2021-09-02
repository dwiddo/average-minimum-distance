from ._Reader import _Reader

try:
    from ccdc.io import EntryReader             # .cif or 'CSD' -> Iterable[Entry]
    from ccdc.search import TextNumericSearch   # refcode -> family of refcodes
    _CCDC_ENABLED = True
except ImportError:
    _CCDC_ENABLED = False

import warnings
def _warning(message, category, filename, lineno, file=None, line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'
warnings.formatwarning = _warning

# Subclasses of _Reader read from different sources (CSD, .cif).
# The __init__ function should initialise self._generator by calling
# self._map(func, iterable) with an iterable

class CSDReader(_Reader):
    """Given CSD refcodes, reads PeriodicSets with ccdc."""
    
    def __init__(self, refcodes, families=False, **kwargs):

        if not _CCDC_ENABLED:
            raise ImportError(f"Failed to import ccdc. Either it is not installed or is not licensed.")

        super().__init__(**kwargs)

        if refcodes is None: 
            families = False
        else:
            refcodes = list(refcodes)
        
        if families:
            
            # extend list of families to all refcodes
            all_refcodes = []
            for refcode in refcodes:
                query = TextNumericSearch()
                query.add_identifier(refcode)
                all_refcodes.extend([hit.identifier for hit in query.search()])

            # filters down to all unique refcodes
            seen = set()
            seen_add = seen.add
            refcodes = [refcode for refcode in all_refcodes if not (refcode in seen or seen_add(refcode))]

        self._generator = self._map(self._Entry_to_PeriodicSet, self._init_ccdc_reader(refcodes))
    
    def _init_ccdc_reader(self, refcodes):
        """Generates ccdc Entries from CSD refcodes"""

        reader = EntryReader('CSD')
        
        if refcodes is None:
            for entry in reader:
                yield entry
        else: 
            for refcode in refcodes:
                try:
                    entry = reader.entry(refcode)
                    yield entry
                except RuntimeError:
                    warnings.warn(f'Identifier {refcode} not found in database')
