from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3.7'
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Libraries',
    'License :: Other/Proprietary License',
]

install_requires = [
	'numpy==1.20.1'
	'scipy==1.6.1',
	'ase>=3.22.0',
	'numba>=0.53.1',
	'h5py>=3.3.0',
]

extras_require = {
    'ccdc': [
        'csd-python-api',
    ],
}

project_urls = {
	'Bug Tracker': 'https://github.com/dwiddo/average-minimum-distance/issues',
	'Changelog': 'https://github.com/dwiddo/average-minimum-distance/blob/master/CHANGELOG.md',
}

kw = {
    'name':             			 'average-minimum-distance',
    'version':          			 '1.1.7',
    'description':      			 'For calculation and comparison of AMD/PDD isometric invariants of periodic sets. Includes .cif reading functionality.',
    'long_description': 			 open('README.md').read(),
    'long_description_content_type': 'text/markdown',
    'author':           			 'Daniel Widdowson',
    'author_email':     			 'sgdwiddo@liverpool.ac.uk',
    'license':          			 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License',
    'keywords':         			 'isometry, invariant, cif, crystal, amd, pdd, similarity, average, minimum, distance, point-wise, distribution',
    'url':              			 'https://github.com/dwiddo/average-minimum-distance',
    'project_urls':					 project_urls,
    'classifiers':      			 classifiers,
    'python_requires':				 '>=3.6',
    'install_requires': 			 install_requires,
    'extras_require':   			 extras_require,
    'packages':         			 find_packages(),
    # 'zip_safe':       			   True,
}

if __name__ == '__main__':
    setup(**kw)