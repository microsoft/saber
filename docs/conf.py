# Configuration file for the Sphinx documentation builder.

import sys
import os

# Add the project root to the path so we can import saber
sys.path.insert(0, os.path.abspath('..'))

from saber._version import __version__

project = 'SABER'
copyright = '2026, Mingqian Feng, Xiaodong Liu, Weiwei Yang'
author = 'Mingqian Feng, Xiaodong Liu, Weiwei Yang'

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google/NumPy docstring support
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# MyST parser settings (for markdown support)
myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
