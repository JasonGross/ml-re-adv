import numpy as np
from inspect_ai.scorer import Metric, Score, mean, metric, std, stderr

from ml_re_adv.utils.data import data_summary_percentiles

# avoid nouse warning
_ = (mean, std, stderr)


@metric
def median() -> Metric:
    """Compute meadian of all scores.

    Returns:
       meadian metric
    """

    def metric(scores: list[Score]) -> float:
        return np.median([score.as_float() for score in scores]).item()

    return metric


@metric
def min() -> Metric:
    """Compute minimum of all scores.

    Returns:
       min metric
    """

    def metric(scores: list[Score]) -> float:
        return np.amin([score.as_float() for score in scores]).item()

    return metric


@metric
def max() -> Metric:
    """Compute maximum of all scores.

    Returns:
       max metric
    """

    def metric(scores: list[Score]) -> float:
        return np.amax([score.as_float() for score in scores]).item()

    return metric


@metric
def sum() -> Metric:
    """Compute sum of all scores.

    Returns:
       sum metric
    """

    def metric(scores: list[Score]) -> float:
        return np.sum([score.as_float() for score in scores]).item()

    return metric


@metric
def prod() -> Metric:
    """Compute product of all scores.

    Returns:
       product metric
    """

    def metric(scores: list[Score]) -> float:
        return np.prod([score.as_float() for score in scores]).item()

    return metric


@metric
def range() -> Metric:
    """Compute range of all scores.

    Returns:
       range metric
    """

    def metric(scores: list[Score]) -> float:
        return np.ptp([score.as_float() for score in scores]).item()

    return metric


@metric
def mean_of_square() -> Metric:
    """Compute mean of the squares of all scores.

    Returns:
       mean of square metric
    """

    def metric(scores: list[Score]) -> float:
        return np.mean([score.as_float() ** 2 for score in scores]).item()

    return metric


_, (
    LowerWhiskerBottomEnd,
    LowerWhiskerCrosshatch,
    QuartileOne,
    Median,
    QuartileThree,
    UpperWhiskerCrosshatch,
    UpperWhiskerTopEnd,
) = data_summary_percentiles()


@metric
def percentile(percentile: float) -> Metric:
    """Compute the percentile of all scores.

    Args:
        percentile (float): The percentile to calculate.

    Returns:
       percentile metric
    """

    def metric(scores: list[Score]) -> float:
        return np.percentile([score.as_float() for score in scores], percentile).item()

    return metric
