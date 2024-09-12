# Pytest MPI Plugin

This pytest plugin provides support for running tests with MPI (Message Passing Interface) using the `mpi4py` library. It allows you to run specific tests under MPI conditions and handles test collection and reporting based on MPI configuration.

## Features

- **MPI Test Selection**: Only runs tests marked with `@pytest.mark.mpi` if MPI support is enabled.
- **Reporting**: Disables output on non-master MPI ranks to reduce noise.
- **Communicator Fixture**: Provides an `communicator` fixture to access the MPI communicator within tests.

## Installation

To use this plugin, you need to have `pytest`, `mpi4py`, and an MPI implementation installed. You can install the required Python package with:

```bash
pip install pytest-fastmpi
```

## Usage

### Enabling MPI tests
To enable MPI tests, use the `--mpi` command-line option when running `pytest`:
```bash
mpiexec -np 4 pytest --mpi
```
This will only run the MPI enabled tests

### Marking MPI tests
You can mark tests with the `@pytest.mark.mpi` decorator and optionally specify the number of MPI ranks with `np`:
```python
import pytest

@pytest.mark.mpi(np=4)
def test_mpi_function(communicator):
    # Your test code here
    pass
```

### Communicator fixture
The `communicator` fixture provides access to the MPI communicator. You can use it in your tests to perform MPI operations. Ensure your test is marked with `@pytest.mark.mpi` to use this fixture.

Example test using the `communicator` fixture:

```python
def test_communicator(communicator):
    rank = communicator.Get_rank()
    size = communicator.Get_size()
    assert size > 1  # Ensure there is more than one MPI process
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue if you find any bugs or have suggestions for improvements.

