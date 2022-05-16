from setuptools import setup, find_packages

description = (
    'Implements fingerprints (isometry invariants) of crystals '
    'based on geometry: average minimum distances (AMD) and '
    'point-wise distance distributions (PDD). '
    'Includes tools to read crystals from files or the CSD.'
)

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

install_requires = [
	'numpy>=1.21',
    'numba>=0.55.0',
	'scipy>=1.6.1',
	'ase>=3.22.0',
	'h5py>=3.3.0',
]

extras_require = {
    'ccdc': [
        'csd-python-api',
    ],
}

project_urls = {
    'Source Code': 'https://github.com/dwiddo/average-minimum-distance/',
    'Documentation': 'https://average-minimum-distance.readthedocs.io/en/latest/',
	'Changelog': 'https://github.com/dwiddo/average-minimum-distance/blob/master/CHANGELOG.md',
}

kw = {
    'name':             			 'average-minimum-distance',
    'version':          			 '1.2.1',
    'description':      			 description,
    'long_description': 			 open('README.md').read(),
    'long_description_content_type': 'text/markdown',
    'author':           			 'Daniel Widdowson',
    'author_email':     			 'sgdwiddo@liverpool.ac.uk',
    'license':          			 license,
    'keywords':         			 keywords,
    'url':              			 'https://github.com/dwiddo/average-minimum-distance',
    'project_urls':					 project_urls,
    'classifiers':      			 classifiers,
    'python_requires':				 '>=3.7',
    'install_requires': 			 install_requires,
    'extras_require':   			 extras_require,
    'packages':         			 find_packages(),
    'include_package_data':          True,
}

if __name__ == '__main__':
    setup(**kw)