# Minimal copy of docsrc/source/conf.py's autodoc settings (no myst_nb), to
# render the API pages that wave 4 touched.
project = "PyBADS API check"
extensions = ["sphinx.ext.autodoc", "numpydoc"]
numpydoc_show_class_members = False
autodoc_default_options = {
    "members": "var1, var2",
    "special-members": "__call__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
