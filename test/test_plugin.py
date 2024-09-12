import pytest


@pytest.mark.mpi
def test_mpi(pytester, communicator):
    """Make sure that our plugin works."""

    # create a temporary pytest test file
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi
        def test_mpi(communicator):
            assert communicator.size > 0
    """
    )

    # run all tests with pytest
    result = pytester.runpytest("--mpi")

    if communicator.rank == 0:
        # check that all 2 tests passed
        result.assert_outcomes(passed=1)


@pytest.mark.mpi
def test_mpi_comm_size(pytester, communicator):
    """Make sure that our plugin works."""

    # create a temporary pytest test file
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi(np=1)
        def test_mpi_one(communicator):
            assert communicator.size == 1

        @pytest.mark.mpi(np=2)
        def test_mpi_two(communicator):
            assert communicator.size == 2

        @pytest.mark.mpi(np=3)
        def test_mpi_three(communicator):
            assert communicator.size == 3
    """
    )

    # run all tests with pytest
    result = pytester.runpytest("--mpi")

    if communicator.rank == 0:
        # check that all 2 tests passed
        result.assert_outcomes(passed=3)

@pytest.mark.mpi
def test_mpi_fail(pytester, communicator):
    """Make sure that our plugin works."""

    # create a temporary pytest test file
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi(np=2)
        def test_mpi_fail_master(communicator):
            assert communicator.rank != 0
        
        @pytest.mark.mpi(np=2)
        def test_mpi_non_master(communicator):
            assert communicator.rank == 0

        @pytest.mark.mpi(np=2)
        def test_mpi_all(communicator):
            assert False
    """
    )

    # run all tests with pytest
    result = pytester.runpytest("--mpi")

    if communicator.rank == 0:
        # check that all 2 tests passed
        result.assert_outcomes(failed=3)




def test_nompi(pytester):
    """Make sure that our plugin works."""

    # create a temporary pytest test file
    pytester.makepyfile(
        """
        import pytest

        def test_non_mpi():
            print("Non-mpi")
            assert True
    """
    )

    # run all tests with pytest
    result = pytester.runpytest()

    # check that all 2 tests passed
    result.assert_outcomes(passed=1)
