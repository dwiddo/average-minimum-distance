#!/usr/bin/env python3
"""average-minimum-distance: isometrically invariant crystal fingerprints

Implements fingerprints (isometry invariants) of crystals
based on geometry: average minimum distances (AMD) and
point-wise distance distributions (PDD). Includes tools to
read crystals from files or the CSD.
"""

import re
from setuptools import setup, find_packages


license = 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License'

keywords = (
    'isometry, invariant, crystal, amd, pdd, similarity, average, '
    'minimum, point-wise, distance, cif'
)

classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: Other/Proprietary License',
    'Natural Language :: English',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3 :: Only',
]

project_urls = {
    'Source Code': 'https://github.com/dwiddo/average-minimum-distance/',
    'Documentation': 'https://average-minimum-distance.readthedocs.io/en/latest/',
    'Changelog': 'https://github.com/dwiddo/average-minimum-distance/blob/master/CHANGELOG.md',
}

description = " ".join(__doc__.split('\n')[2:])

with open(r'requirements.txt') as f:
    install_requires = [req.strip() for req in f.readlines()]

with open(r'amd/__init__.py') as f:
    version = re.search("__version__ = '(.*)'", f.read()).group(1)

kw = {
    'name': 'average-minimum-distance',
    'version': version,
    'description': description,
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
    'extras_require': {'ccdc': ['csd-python-api']},
    'packages': find_packages(),
    'include_package_data': True,
}

if __name__ == '__main__':
    setup(**kw)
