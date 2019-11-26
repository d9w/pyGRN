# pyGRN

[![Build Status](https://travis-ci.org/d9w/pyGRN.svg?branch=master)](https://travis-ci.org/d9w/pyGRN)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![codecov](https://codecov.io/gh/d9w/pyGRN/branch/master/graph/badge.svg)](https://codecov.io/gh/d9w/pyGRN)

Artificial Gene Regulatory Network model and evolution in Python

## Installation

Recommended installation method uses python's venv module:

```bash
git clone https://github.com/d9w/pyGRN.git
cd pyGRN
python -m venv [VENV_PATH]
source [VENV_PATH]/bin/activate
pip install -e .
```

where `VENV_PATH` is determined by the user (for example, `~/.venvs/pygrn`)

`pyGRN` requires a `grns` directory and a `logs` directory. By default, these
will be in the same directory that scripts are run from; for example:

```bash
cd pyGRN
mkdir grns
mkdir logs
```

GRN files saved in JSON format will be generated during evolution and placed in
`grns`, while evolutionary logs will be written in `logs`.

## Usage

Examples of usage can be found in `regression.py`, which can run various
regression problems, and `dqn.py`, which
uses [keras-rl](https://github.com/keras-rl/keras-rl) to perform deep Q learning
with a GRN.

## Testing

To run the tests, use `pytest` or `pytest --cov=./` to get a coverage report.
Certain tests are stochastic and should pass most of the time, but might fail
simply due to random initialization (eg `tests/grns/test_grn_compare.py`). To
run these tests with verbose output, use:

```
pytest -v -s tests/grns/test_grn_compare.py
```

The tests also have many examples of usage beyond the above files.
