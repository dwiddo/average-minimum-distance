# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath(r'../..'))

import mock
 
MOCK_MODULES = [
    'numba',
    'scipy',
    'scipy.spatial',
    'scipy.special',
    'scipy.spatial.distance',
    'scipy.sparse',
    'scipy.sparse.csgraph',
    'ase',
    'ase.io.cif',
    'ase.spacegroup.spacegroup',
    'h5py', 
    'ccdc',
    'ccdc.io',
    'ccdc.search'
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# -- Project information -----------------------------------------------------

project = 'average-minimum-distance'
copyright = '2022, Daniel Widdowson'
author = 'Daniel Widdowson'

# The full version, including alpha/beta/rc tags
release = '1.1.8'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.coverage', 
    'sphinx.ext.napoleon',
    # 'sphinx.ext.autosummary',
    # 'rinoh.frontend.sphinx',
    'sphinx_rtd_dark_mode',
    'm2r2',
]

# autodoc_typehints = "description"
source_suffix = ['.rst', '.md']

# Puts functions in order of source instead of alphabetical
autodoc_member_order = 'bysource'
# autodoc_default_flags = ['members']
# autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_rtd_theme'
default_dark_mode = True

html_sidebars = { 
    '**': [
        'globaltoc.html', 
        'relations.html', 
        'sourcelink.html', 
        'searchbox.html'
        ] 
}

# # Add any paths that contain custom static files (such as style sheets) here,
# # relative to this directory. They are copied after the builtin static files,
# # so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']