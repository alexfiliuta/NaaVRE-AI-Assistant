"""Regression from live experiments: decoding CSV rows is part of reading a CSV."""
import pytest
from naavre_ai_assistant.llm_service import _has_combined_actions


@pytest.mark.xfail(strict=True,reason='Current wording heuristic treats CSV decoding to string dictionaries as a separate preparation job and rejects a legitimate single read job.')
def test_csv_decoding_is_not_a_second_scientific_job():
    assert not _has_combined_actions('Read the CSV file and parse each row into a string-valued dictionary using csv.DictReader.')


@pytest.mark.parametrize('purpose',[
    'Read the CSV and validate all values.',
    'Read the CSV and convert height_cm strings into numeric values.',
    'Plot the results and export the chart file.',
])
def test_actual_distinct_phases_remain_separate(purpose):
    assert _has_combined_actions(purpose)
