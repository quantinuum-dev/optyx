# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from optyx import __version__ as version, __version_info__ as v
trim_version = f'{v[0]}.{v[1]}.{v[2]}'
if version.startswith(f'{trim_version}.'):
    version = f'{v[0]}.{v[1]}.{int(v[2]) - 1} [git latest]'
release = version


project = 'optyx'
copyright = '2023, Quantinuum Ltd.'
author = 'Quantinuum Ltd.'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_parser',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
]

autosummary_generate = True

autodoc_mock_imports = ["pytket", "pennylane", "torch", "sympy"]

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = False

napoleon_use_admonition_for_examples = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'discopy': ("https://docs.discopy.org/en/main/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_context = {
  'display_github': True,
  'github_user': 'CQCL',
  'github_repo': 'optyx',
  'github_version': 'main',
  'conf_py_path': '/docs/'
}

html_static_path = ['_static']
