import numpy as np
import pytest
from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    # Generate random data
    X = np.random.normal(0, 1, 1000)

    # Initialize the Winsorizer
    winsorizer = Winsorizer(lower_quantile=lower_quantile, upper_quantile=upper_quantile)

    # Fit the Winsorizer
    winsorizer.fit(X)

    # Check that the calculated quantiles are correct
    expected_lower = np.quantile(X, lower_quantile)
    expected_upper = np.quantile(X, upper_quantile)

    assert winsorizer.lower_quantile_ == pytest.approx(expected_lower, rel=1e-6)
    assert winsorizer.upper_quantile_ == pytest.approx(expected_upper, rel=1e-6)

    # Transform the data
    X_transformed = winsorizer.transform(X)

    # Ensure that all values are within the computed quantile bounds
    assert np.all(X_transformed >= winsorizer.lower_quantile_)
    assert np.all(X_transformed <= winsorizer.upper_quantile_)

    # Additional test for edge cases
    if lower_quantile == upper_quantile:
        # If lower and upper quantiles are the same, all values should equal that quantile
        assert np.all(X_transformed == expected_lower)
