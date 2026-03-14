project = "tract-python"
copyright = "Sonos"
author = "Sonos"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

html_theme = "furo"
autodoc_member_order = "bysource"
napoleon_google_docstring = True

master_doc = "docs/index"

exclude_patterns = [
    "_build",
    ".venv",
    ".pytest_cache",
    "rust-workspace",
    "tract.egg-info",
    "tests",
    "setup.py",
]
