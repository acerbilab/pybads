# BADS subpackage developing notes

These notes are used for keeping track of ToDos and porting information.

* Support for periodic variables
* Benchmark PyBADS on cognitive and neural science models ([neurobench](https://github.com/lacerbi/neurobench))

DONE:
- snake case for advanced_bads_options
- Can you please go through the "for developers" page and update it as appropriate (it is a bit out of date). Use as reference the updated "for developers" PyVBMC page.
- Can you also do a search through the code to check references to lb/ub/plb/pub and change them to the extended version (lower_bounds, etc.)? I saw that they might appear in some warning/error messages. Don't do it blindly with "replace all": sometimes (e.g., in the BADS docstring) we are okay using plb/pub/lb/ub on purpose as a shorthand, which we define in that section (since we keep repeating them in the docstrings).
- Upload BADS cartoon in PyBADS
