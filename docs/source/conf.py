# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import re
import sys
import mock

sys.path.insert(0, os.path.abspath(r"../../src"))

# -- Project information -----------------------------------------------------

project = "average-minimum-distance"
copyright = "2025, Daniel Widdowson"
author = "Daniel Widdowson"

with open(r"../../src/amd/__init__.py") as f:
    version = re.search('__version__ = "(.*)"', f.read()).group(1)

for mod_name in ["ccdc", "ccdc.io", "ccdc.search"]:
    sys.modules[mod_name] = mock.Mock()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "sphinx_rtd_dark_mode",
    "sphinx_mdinclude",
]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    # 'numba':    ('https://numba.pydata.org/numba-doc/latest/', None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "gemmi": ("https://gemmi.readthedocs.io/en/latest/", None),
    "pymatgen": ("https://pymatgen.org/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    # 'tqdm':     ('https://tqdm.github.io/', None),
    "ccdc": ("https://downloads.ccdc.cam.ac.uk/documentation/API/", None),
}

# autodoc_typehints = "description"
source_suffix = [".rst", ".md"]

# Puts functions in order of source instead of alphabetical
autodoc_member_order = "bysource"
# autodoc_default_flags = ['members']
# autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_rtd_theme"
default_dark_mode = True

html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}

# # Add any paths that contain custom static files (such as style sheets) here,
# # relative to this directory. They are copied after the builtin static files,
# # so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
