from setuptools import setup, find_packages

setup(name='pybads',
      version='0.0.1',
      description='Bayesian Adaptive Direct Search',
      packages=find_packages(),
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'pytest',
                        'sphinx',
                        'numpydoc',
                        'cma',
                        'corner',

                        'imageio',
                        'kaleido',

                        'myst_nb',
                        'numpydoc',
                        'sphinx',
                        'sphinx-book-theme',
                        'pylint',
                        'pytest',
                        'pytest-cov',
                        'pytest-mock',
                        'pytest-rerunfailures'],
     )
