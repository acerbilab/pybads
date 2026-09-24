# Makefile for extracting code from Jupyter Notebooks: each script is the
# code of the notebook of the same stem in ../, without its markdown cells.
# Needs GNU Make, and nbconvert, black and isort (at the pre-commit hook
# versions) in the environment that `python` names.

NOTEBOOKS = $(wildcard ../pybads_example_*.ipynb)
SCRIPTS = $(patsubst ../%.ipynb,%.py,$(NOTEBOOKS))

all: $(SCRIPTS)

%.py: ../%.ipynb
  # convert NB to script | remove leading lines | remove trailing line > output
	@python -m nbconvert $< --no-prompt --stdout --to script --PythonExporter.exclude_markdown=True | sed -e '1,3d' | head -n -1 > $@
	@python -m isort $@
	@python -m black $@

clean:
	@rm -f pybads_example_*.py

.PHONY: all clean
