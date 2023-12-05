# Contributing guidelines

This project use [`tox`](https://pypi.org/project/tox/) to simplify virtual environment management and testing.

## Dependencies
Dependencies are defined in `pyproject.toml` using version ranges. 
We want to define broad version ranges so the users of this library won't run into dependency conflict.

### Pinned dependencies
Pinned versions of dependencies are defined in `requirements.txt` and `requirements-test.txt`. 
These are used for local development and testing on CI.
These files are automatically generated using `pip-compile` TOX environment.

To re-generate requirements files run:
```
tox run -e pip-compile
```

## Running tests
You can run tests for all supported Python versions using:
```
tox run
```

Or for specific Python version:
```
tox run -e py311
```

Or manually invoke pytest:
```
pytest tests
```

## Build
Build tools are defined in `pyproject.toml`.
Backend tool is [`hatchling`](https://hatch.pypa.io/latest/) - responsible for building distribution package.
CLI tool is [`build`](https://pypa-build.readthedocs.io/en/stable/index.html).
The package name specified in .toml config must match the directory name src/[package_name]
if `package_name` contains `__init__.py`.

You can run the build manually:
1. Install build: `python -m pip install build`
2. Run `python -m build` from the project root - this will create wheel in the `/dist` folder.

Or using tox:
1. Run `tox run`
2. Wheel is created in `.tox/.pkg/dist` folder.