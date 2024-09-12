import pytest

## Defines
MPI_ARG = "--mpi"

class MPIRunnerPlugin:
    def pytest_collection_modifyitems(self, config, items):
        enabled = config.getoption(MPI_ARG)
        for item in items:
            if "mpi" in item.keywords and not enabled:
                item.add_marker(pytest.mark.skip(reason="MPIRunner not enabled"))
            elif not "mpi" in item.keywords and enabled:
                item.add_marker(pytest.mark.skip(reason="Skipping non-mpi test"))

    def manage_reporting(self, config):
        if config.getoption(MPI_ARG):
            try:
                from mpi4py import MPI
            except ImportError:
                return

            if MPI.COMM_WORLD.rank != 0:
                # Disable output on all non-master ranks
                config.pluginmanager.set_blocked(name="terminalreporter")

def pytest_configure(config):
    config.addinivalue_line("markers", "mpi(np): Run test with n amout of mpi ranks")

    plugin = MPIRunnerPlugin()
    config.pluginmanager.register(plugin)
    plugin.manage_reporting(config)

def pytest_addoption(parser):
    group = parser.getgroup("mpi", description="support for MPI-enabled code")
    group.addoption(MPI_ARG, action="store_true", default=False, help="Run MPI tests")

@pytest.fixture
def communicator(request):
    from mpi4py import MPI

    mpi_marker = request.node.get_closest_marker("mpi")
    if not mpi_marker:
        raise RuntimeError("No MPI marker found!")

    comm_size = mpi_marker.kwargs.get("np")

    comm = MPI.COMM_WORLD

    size = comm.size
    rank = comm.rank

    comm.Barrier()

    if comm_size is None:
        yield comm
    else:
        if comm_size > size:
            raise RuntimeError(f"np ({comm_size}) must be smaller than the total amount of mpi ranks ({size})")

        my_comm = comm.Split(color=rank<comm_size, key=rank)

        if (rank >= comm_size):
            comm.Barrier()
            pytest.skip("MPI-rank not required")

        yield my_comm

        my_comm.Free()

    comm.Barrier()

