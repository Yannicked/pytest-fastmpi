import pytest
import dataclasses
from typing import Any, Optional
from _pytest.stash import StashKey
from _pytest.outcomes import skip
from _pytest.config import hookimpl
from mpi4py import MPI
import os
import sys

## Defines
MPI_ARG = "--mpi"


@dataclasses.dataclass(frozen=False)
class MPIRunner:
    __slots__ = ("active", "communicator")

    active: bool

    communicator: Any


MPIRunner_key = StashKey[Optional[MPIRunner]]()


class MPIRunnerPlugin:
    def pytest_collection_modifyitems(self, config, items):
        enabled = config.getoption(MPI_ARG)
        for item in items:
            if "mpi" in item.keywords and not enabled:
                item.add_marker(pytest.mark.skip(reason="MPIRunner not enabled"))
            elif "mpi" not in item.keywords and enabled:
                item.add_marker(pytest.mark.skip(reason="Skipping non-mpi test"))

    def manage_reporting(self, config):
        if config.getoption(MPI_ARG):
            try:
                from mpi4py import MPI
            except ImportError:
                return

            if MPI.COMM_WORLD.rank != 0:
                # Disable output on all non-master ranks
                # config.pluginmanager.set_blocked(name="terminalreporter")
                # Very hacky
                sys.stdout = open(os.devnull, "w")
                config.workerinput = None
                config.pluginmanager.set_blocked(name="_cov")

    def create_communicator(self, size):
        comm = MPI.COMM_WORLD
        rank = comm.rank

        split_rank = comm.Split(color=rank < size, key=rank)
        return split_rank

    @hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        comm = MPI.COMM_WORLD
        size = comm.size
        rank = comm.rank

        mpi_marker = item.get_closest_marker("mpi")

        if mpi_marker is None:
            return

        comm.Barrier()

        test_size = mpi_marker.kwargs.get("np", size)

        split_communicator = self.create_communicator(test_size)
        active = rank < test_size
        item.stash[MPIRunner_key] = MPIRunner(active, split_communicator)
        if not active:
            raise skip.Exception(
                "MPI rank not required in test", _use_item_location=True
            )

    def gather_results_from_ranks(self, communicator, rep):
        results = communicator.gather(rep, root=0)
        if communicator.rank != 0:
            return rep

        reprs = []
        for rank, result in enumerate(results):
            if result.outcome == "failed":
                rep.outcome = "failed"
                reprs.append(f"===Rank: {rank}===\n{result.longrepr}")
        if rep.outcome == "failed":
            rep.longrepr = "\n\n".join(reprs)
        return rep

    @hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item, call):
        rep = yield
        mpi_data = item.stash.get(MPIRunner_key, None)
        if mpi_data and mpi_data.active:
            rep = self.gather_results_from_ranks(mpi_data.communicator, rep)
        if mpi_data:
            mpi_data.communicator.Barrier()
            if call.when == "teardown":
                mpi_data.communicator.Free()
        return rep


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
    mpi_data = request.node.stash.get(MPIRunner_key, None)
    if not mpi_data:
        raise RuntimeError("No MPIRunner data found in stash")

    return mpi_data.communicator
