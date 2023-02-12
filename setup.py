#!/usr/bin/env python
"""average-minimum-distance: geometry-based crystal fingerprints

Descriptors (isometry invariants) of crystals based on geometry: 
average minimum distances (AMD) and point-wise distance distributions
(PDD).
"""

import re
from setuptools import setup, find_packages


license = 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License'

keywords = (
    'isometry, invariant, descriptor, crystal, amd, pdd, similarity, '
    'average, minimum, pointwise, distance, cif'
)

classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: Other/Proprietary License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Topic :: Scientific/Engineering :: Chemistry',
    'Topic :: Scientific/Engineering :: Mathematics',
]

project_urls = {
    'Source Code': 'https://github.com/dwiddo/average-minimum-distance/',
    'Documentation': 'https://average-minimum-distance.readthedocs.io/en/latest/',
    'Changelog': 'https://github.com/dwiddo/average-minimum-distance/blob/master/CHANGELOG.md',
}

with open(r'requirements.txt') as f:
    install_requires = [req.strip() for req in f.readlines()]

extras_require = {
    'ccdc': ['csd-python-api'],
    'gemmi': ['gemmi'],
    'pymatgen': ['pymatgen'],
    'dev': ['pytest'],
}

with open(r'amd/__init__.py') as f:
    version = re.search("__version__ = '(.*)'", f.read()).group(1)

kw = {
    'name': 'average-minimum-distance',
    'version': version,
    'description': 'Descriptors of crystals based on geometry (isometry invariants).',
    'long_description': open('README.md').read(),
    'long_description_content_type': 'text/markdown',
    'author': 'Daniel Widdowson',
    'author_email': 'D.E.Widdowson@liverpool.ac.uk',
    'license': license,
    'keywords': keywords,
    'url': 'https://github.com/dwiddo/average-minimum-distance',
    'project_urls': project_urls,
    'classifiers': classifiers,
    'python_requires': '>=3.7',
    'install_requires': install_requires,
    'extras_require': extras_require,
    'packages': find_packages(),
    'include_package_data': True,
    'entry_points': {
        'console_scripts': [
            'amd-compare = amd.cli:main',
        ]
    }
}


if __name__ == '__main__':
    setup(**kw)
