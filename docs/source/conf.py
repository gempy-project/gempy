#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Updated conf.py for optimized documentation generation
# FW 11/2024

import datetime
import os
import sys
import warnings
import pyvista
from sphinx_gallery.sorting import FileNameSortKey

# Add directories to sys.path
sys.path.insert(0, os.path.abspath('.'))

# External gallery examples
import make_external_gallery
make_external_gallery.make_example_gallery()

# PyVista configuration
pyvista.set_error_output_file('errors.txt')
pyvista.OFF_SCREEN = True
pyvista.set_plot_theme('document')
pyvista.FIGURE_PATH = os.path.join(os.path.abspath('_images/'), 'auto-generated/')
pyvista.BUILDING_GALLERY = True
os.makedirs(pyvista.FIGURE_PATH, exist_ok=True)

# Project Information
project = 'GemPy'
year = datetime.date.today().year
copyright = f'2017-{year}, GemPy Developers'

import gempy  # Ensure GemPy is accessible
version = gempy.__version__
release = gempy.__version__

# Sphinx Configuration
with open(os.path.join(os.path.dirname(__file__), '../../AUTHORS.rst'), 'r') as f:
    author = f.read()

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
    'pyvista.ext.plot_directive',
    'sphinx_design',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
]

# Optimize Autosummary
autosummary_generate = True  # Ensure this is set only when required
autosummary_imported_members = False  # Disable imported members to reduce rebuilds

# Autodoc Default Options
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}
autodoc_mock_imports = ["numpy", "pandas", "matplotlib", "pyvista"]  # Avoid runtime imports

# Sphinx Gallery Configuration
sphinx_gallery_conf = {
    "examples_dirs": ["../../examples/tutorials", "../../examples/examples"],
    "gallery_dirs": ["tutorials", "examples"],
    "filename_pattern": r"\.py",
    "ignore_pattern": r"(__init__\.py|__init__\.ipynb|ch1_7_3d_visualization\.py|mik\.py)",  # Ignore __init__.ipynb
    # "ignore_pattern": r"__init__.*",
    "abort_on_example_error": False,  # Log errors but continue building
    "within_subsection_order": FileNameSortKey,
    "image_scrapers": ('pyvista', 'matplotlib'),
    "first_notebook_cell": ("%matplotlib inline\nfrom pyvista import set_plot_theme\nset_plot_theme('document')"),
    "reference_url": {
        'gempy': None,
        'numpy': 'https://numpy.org/doc/stable/',
    },
}

# Path Setup
templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store', 'errors.txt', '../test']
source_suffix = ['.rst']
master_doc = 'index'
language = 'en'

# HTML Output Options
html_theme = 'alabaster'
# html_theme = "furo"
html_theme_options = {
    'github_user': 'gempy-project',
    'github_repo': 'gempy',
    'github_type': 'star',
    'logo': 'logos/gempy.png',
    'page_width': '1200px',
    'sidebar_collapse': True,
}
html_static_path = ['_static']
# for dark mode - but needs more testing
# html_css_files = ["dark_mode.css"]
# html_js_files = ["dark_mode.js"]
html_favicon = '_static/logos/favicon.ico'

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'gempy', 'GemPy Documentation', [author], 1)]

# Suppress Warnings
warnings.filterwarnings("ignore", category=UserWarning, message='Matplotlib is currently using agg')
pygments_style = 'sphinx'
highlight_language = 'python3'
todo_include_todos = True

