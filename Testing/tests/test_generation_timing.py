import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('generation_timing', Path(__file__).resolve().parents[1] / 'summarize-generation-timing.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)

LOG = '''[I 2026-10-09 11:45:53.315 ServerApp] NaaVRE local model request started: kind=agent decision model=qwen3.8:27b thinking=False
[I 2026-10-09 11:46:33.880 ServerApp] NaaVRE local model response received: kind=agent decision elapsed=28.5s
[I 2026-10-09 11:48:02.611 ServerApp] NaaVRE planning attempt 1/6 started
[W 2026-10-09 11:48:49.342 ServerApp] NaaVRE plan rejected; retrying attempt 2/6
[I 2026-10-09 11:49:26.824 ServerApp] NaaVRE generating cell 1/6 (attempt 1/3)
[I 2026-10-09 11:49:48.289 ServerApp] NaaVRE local model response received: kind=cell 1 elapsed=21.4s
[I 2026-10-09 11:54:22.996 ServerApp] NaaVRE local model response received: kind=cell 6 elapsed=49.7s
[I 2026-10-09 11:58:16.008 ServerApp] Saving file at /Energy_Test.ipynb'''


def test_energy_log_boundaries_exclude_saving_and_human_gap():
    result = timing.summarize(LOG)
    assert result['build_to_last_model_response_seconds'] == 380.385
    assert result['planning_until_first_cell_seconds'] == 84.213
    assert result['cell_generation_span_seconds'] == 296.172
    assert result['observed_investigation_model_span_seconds'] == 40.565
    assert result['investigation_to_build_gap_seconds'] == 88.731
    assert result['combined_wall_span_including_gap_seconds'] == 509.681
    assert result['execution_time_seconds'] is None


def test_two_builds_are_not_accidentally_merged():
    with pytest.raises(ValueError, match='exactly one build'):
        timing.summarize(LOG + '\n' + LOG)
