import numpy as np

from verl.experimental.fully_async_policy.detach_utils import _normalize_param_versions


def test_initial_rollout_versions_are_treated_as_version_zero():
    param_version_start = np.array([None, 0, 2], dtype=object)
    param_version_end = np.array([None, 1, 2], dtype=object)

    normalized_start = _normalize_param_versions(param_version_start)
    normalized_end = _normalize_param_versions(param_version_end)
    spans = [abs(end - start) for end, start in zip(normalized_end, normalized_start, strict=False)]

    assert normalized_end == [0, 1, 2]
    assert spans == [0, 1, 0]
