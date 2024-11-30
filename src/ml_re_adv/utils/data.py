from typing import Literal, Tuple

from jaxtyping import Float
from numpy import ndarray
from scipy import stats


def data_summary_percentiles() -> Tuple[
    Tuple[
        Literal["LowerWhiskerBottomEnd"],
        Literal["LowerWhiskerCrosshatch"],
        Literal["QuartileOne"],
        Literal["Median"],
        Literal["QuartileThree"],
        Literal["UpperWhiskerCrosshatch"],
        Literal["UpperWhiskerTopEnd"],
    ],
    Float[ndarray, "7"],
]:
    """
    Generate a summary of data percentiles.

    This function calculates the percentiles corresponding to specific
    statistical points in a normal distribution. These points include
    the lower whisker bottom end, lower whisker crosshatch, first quartile,
    median, third quartile, upper whisker crosshatch, and upper whisker top end.

    The percentiles are calculated as follows:
    - LowerWhiskerBottomEnd: The 0.15th percentile, calculated as the cumulative
      distribution function (CDF) value at -3 times the standard deviation.
    - LowerWhiskerCrosshatch: The 2.28th percentile, calculated as the CDF value
      at -2 times the standard deviation.
    - QuartileOne: The 15.87th percentile, calculated as the CDF value at -1 times
      the standard deviation.
    - Median: The 50th percentile, calculated as the CDF value at 0 times the
      standard deviation.
    - QuartileThree: The 84.13th percentile, calculated as the CDF value at 1 times
      the standard deviation.
    - UpperWhiskerCrosshatch: The 97.72th percentile, calculated as the CDF value
      at 2 times the standard deviation.
    - UpperWhiskerTopEnd: The 99.85th percentile, calculated as the CDF value at
      3 times the standard deviation.

    Returns:
        Tuple[
            Tuple[
                Literal["LowerWhiskerBottomEnd"],
                Literal["LowerWhiskerCrosshatch"],
                Literal["QuartileOne"],
                Literal["Median"],
                Literal["QuartileThree"],
                Literal["UpperWhiskerCrosshatch"],
                Literal["UpperWhiskerTopEnd"],
            ],
            Float[ndarray, "7"],
        ]: A tuple containing a tuple of literal names for the percentiles and
        an array of the corresponding percentile values.
    """
    s = _twenty_five_percent_in_std_dev = stats.norm.ppf(0.75)
    percentiles = stats.norm.cdf([-3 * s, -2 * s, -s, 0, s, 2 * s, 3 * s])
    return (
        "LowerWhiskerBottomEnd",
        "LowerWhiskerCrosshatch",
        "QuartileOne",
        "Median",
        "QuartileThree",
        "UpperWhiskerCrosshatch",
        "UpperWhiskerTopEnd",
    ), percentiles
