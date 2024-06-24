# PyPulCeq

Python tools to write Pulseq files for GEHC scanners (i.e., [TOPPE interpreter](https://toppemri.github.io/)).

## Installation

The ``PyPulCeq`` package can be installed from source using pip:


1. Clone the repo

  ```bash
  git clone git@github.com:INFN-PREDATOR/pypulceq.git
  ```

2. Navigate to the repository root folder and install using pip:

  ```bash
   pip install .
  ```

## Usage

PyPulCeq can be used as a library in [Pypulseq](https://github.com/imr-framework/pypulseq/tree/dev) -based sequence design routines as follows:

```python
import pypulceq
seq = my_design_routine() # replace with your design

# create GEHC system specs 
# if not provided, parse from pypulseq.Opts in seq.system,
# and use default nslices=2048, nviews=600 for frame indexing
sys = pypulceq.SystemSpecs()

# perform actual conversion
pypulceq.seq2ge("myseq", seq, sys)
```

In addition, we provide a Command Line Interface for conversion of pre-generated ``.seq`` files:

```bash
seq2ge --output-name "myseq" --input-file "pypulseq_file.seq" --nviews NPHASEENC --nslices NSLICES
```

An in-browser executable example can be found in the [examples](https://github.com/INFN-PREDATOR/pypulceq/blob/main/examples/getting_started.ipynb) folder. You can try it in Colab: <a target="_blank" href="https://colab.research.google.com/github/https://github.com/INFN-PREDATOR/pypulceq/blob/main/examples/getting_started.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Developers

Developers can install ``PyPulCeq`` in editable mode and include the additional dependencies (``black, pytest``):

```
pip install -e .[dev]
```

Tests can be executed from ``PyPulCeq`` root folder using ``pytest``:

```
pytest .
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`pypulceq` was created by Matteo Cencini. It is licensed under the terms of the MIT license.

## Credits

`pypulceq` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter). Code is based on the original MATLAB version from [HarmonizedMRI](https://github.com/HarmonizedMRI/PulCeq/tree/main).
