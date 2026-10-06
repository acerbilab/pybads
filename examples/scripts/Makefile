# Makefile for extracting code from Jupyter Notebooks: each script is the
# code of the notebook of the same stem in ../, without its markdown cells.
# Needs GNU Make run from a POSIX shell (Git Bash on Windows), and nbconvert,
# IPython (without which nbconvert leaves the code as it is and the last line
# of the script is cut), black and isort (at the pre-commit hook versions)
# in the environment that `python` names.
#
# `make run` executes the notebooks in place, from any shell and with
# ipykernel too, on the PyBADS and gpyreg that `python` imports. It prints
# first their locations and the versions of their installed metadata, which
# for PyBADS is what a result's `version` reports, and the commit of a
# gpyreg that is a git checkout: gpyreg's metadata can belong to another
# installation than the one imported, such as an editable install made at
# another commit, or one that `PYTHONPATH` shadows. As Jupyter's interface
# does, it merges the consecutive pieces of a cell's output stream, which
# nbconvert alone saves one for each flush of the kernel's stream, and it
# leaves the times of the execution out of the cells' metadata.

NOTEBOOKS = $(wildcard ../pybads_example_*.ipynb)
SCRIPTS = $(patsubst ../%.ipynb,%.py,$(NOTEBOOKS))

all: $(SCRIPTS)

%.py: ../%.ipynb
  # convert NB to script | remove leading lines | remove trailing line > output
	@python -m nbconvert $< --no-prompt --stdout --to script --PythonExporter.exclude_markdown=True | sed -e '1,3d' | head -n -1 > $@
	@python -m isort $@
	@python -m black $@

# The outputs that ship with a release show no reminder of an old release
run: export PYBADS_NO_UPDATE_REMINDER = 1
run:
	@python -c "import os, shutil, subprocess, gpyreg, pybads; from importlib.metadata import version as v; print('PyBADS', v('pybads'), pybads.__file__); print('gpyreg', v('gpyreg'), gpyreg.__file__); d = os.path.dirname(os.path.dirname(gpyreg.__file__)); r = subprocess.run(['git', '-C', d, 'describe', '--tags', '--always', '--dirty'], capture_output=True, text=True) if os.path.exists(os.path.join(d, '.git')) and shutil.which('git') else None; r and r.returncode == 0 and print('gpyreg checkout at', r.stdout.strip())"
  # a preprocessor named here runs after the execution, the default one before
	@python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.record_timing=False --NotebookExporter.preprocessors=nbconvert.preprocessors.CoalesceStreamsPreprocessor $(NOTEBOOKS)

clean:
	@rm -f pybads_example_*.py

.PHONY: all clean run
