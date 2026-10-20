import json

import pytest

from naavre_ai_assistant.input_schema import inspect_table, schema_column_error, SAMPLE_ROWS
from naavre_ai_assistant.llm_service import LLMCodeGenerator, _file_aware_context
from test_llm_service import config, FakeClient, valid_analysis
from test_context_and_files import oversized_notebook


HEADERS = ['timestamp', 'station_id', 'temperature_c', 'salinity_psu', 'depth_m']


@pytest.fixture
def observations(tmp_path):
    path = tmp_path / 'observations.csv'
    path.write_text(','.join(HEADERS) + '\n2026-01-01T00:00:00,ST-01,16.95,35.0,10.0\n', encoding='utf-8-sig')
    return path


def test_inspection_reads_real_headers_including_units_and_sample_types(observations):
    original = observations.read_bytes()
    schema = inspect_table(str(observations))
    assert schema['headers'] == HEADERS
    assert schema['columns'][0]['sample_types'] == ['ISO timestamp']
    assert schema['columns'][1]['sample_types'] == ['text']
    assert schema['columns'][2]['sample_types'] == ['number']
    assert schema['sample_rows'] == 1
    assert observations.read_bytes() == original


@pytest.mark.parametrize('extension,separator', [('.csv', ';'), ('.tsv', '\t')])
def test_delimiters_and_quoted_headers(tmp_path, extension, separator):
    path = tmp_path / f'values{extension}'
    path.write_text(f'"time label"{separator}"temperature, C"\n2026-01-01{separator}22\n')
    schema = inspect_table(str(path))
    assert schema['headers'] == ['time label', 'temperature, C']
    assert schema['delimiter'] == separator


@pytest.mark.parametrize('header', ['a,a', 'a,'])
def test_bad_headers_reported_before_model_call(tmp_path, header):
    path = tmp_path / 'bad.csv'
    path.write_text(header + '\n1,2\n')
    with pytest.raises(ValueError, match='column names'):
        inspect_table(str(path))


def test_sampling_is_bounded_and_does_not_invent_whole_file_statistics(tmp_path):
    path = tmp_path / 'large.csv'
    path.write_text('x\n' + '1\n' * 1000)
    schema = inspect_table(str(path))
    assert schema['sample_rows'] == SAMPLE_ROWS
    assert schema['sample_only'] is True


def test_schema_survives_oversized_notebook_context(observations):
    context = json.loads(_file_aware_context(oversized_notebook(), f'Read {observations.as_posix()}', {}, {}))
    assert context['file_references'][0]['schema']['headers'] == HEADERS
    assert len(json.dumps(context, indent=2, sort_keys=True)) < 50_000


def test_two_phases_keep_actual_schema_over_wrong_prompt_and_retry_bad_columns(observations):
    bad_code = "required_columns = ['station_id', 'date', 'temperature']\nfor row in data:\n    print(row['date'])"
    good_code = "for row in data:\n    print(row['timestamp'], row['temperature_c'])"
    client = FakeClient(valid_analysis(), bad_code, good_code)
    service = LLMCodeGenerator(config(), client)
    prompt = f'Read {observations.as_posix()}. Columns are date, station and temperature.'
    analysis = service.analyze_request(prompt)
    assert analysis['input_schemas'][0]['headers'] == HEADERS
    code = service.generate_code_from_analysis(prompt, analysis=analysis)
    assert code == good_code
    assert len(client.completions.requests) == 3
    for request in client.completions.requests:
        assert all(column in request['messages'][1]['content'] for column in HEADERS)
    assert 'Unverified input columns' in client.completions.requests[-1]['messages'][1]['content']


def test_every_planned_cell_gets_the_same_schema(observations):
    plan = json.loads(valid_analysis())
    plan['cell_plan'] = [dict(plan['cell_plan'][0], title=f'Job {i}') for i in range(3)]
    client = FakeClient(json.dumps(plan), 'data=[]', 'result=[]', 'print(result)')
    service = LLMCodeGenerator(config(), client)
    result = service.generate_cells(f'Inspect {observations.as_posix()}')
    assert len(result) == 3
    for request in client.completions.requests[1:]:
        assert 'Authoritative input schemas' in request['messages'][1]['content']
        assert all(column in request['messages'][1]['content'] for column in HEADERS)


def test_file_change_between_phases_requires_new_plan(observations):
    client = FakeClient(valid_analysis())
    service = LLMCodeGenerator(config(), client)
    prompt = f'Read {observations.as_posix()}'
    analysis = service.analyze_request(prompt)
    observations.write_text('new_column\n42\n')
    with pytest.raises(ValueError, match='changed since planning'):
        service.generate_code_from_analysis(prompt, analysis=analysis)
    assert len(client.completions.requests) == 1


def test_derived_fields_are_allowed_but_raw_input_column_typos_are_not(observations):
    schemas = [inspect_table(str(observations))]
    assert schema_column_error("x = row.get('date')", schemas)
    assert schema_column_error("x = row['temperature']", schemas)
    assert schema_column_error("x = row['temperature_c']", schemas) is None
    assert schema_column_error("x = row['date']", schemas, "converted = [{'date': row['timestamp']} for row in data]") is None
    assert schema_column_error("for r in data:\n    print(r['date'])", schemas)
    assert schema_column_error("required_columns = ['date']", schemas, "converted = [{'date': row['timestamp']} for row in data]")


def test_computed_aggregate_fields_are_allowed_but_not_raw_header_aliases(observations):
    schemas = [inspect_table(str(observations))]
    prior = (
        "for row in data:\n"
        "    groups[row['station_id']].append(float(row['temperature_c']))\n"
        "aggregated = []\n"
        "for station, values in groups.items():\n"
        "    aggregated.append({'station_id': station, 'timestamp': timestamp, "
        "'mean_temperature_c': sum(values) / len(values)})"
    )
    assert schema_column_error(
        "for row in aggregated:\n    print(row['mean_temperature_c'])", schemas, prior
    ) is None
    assert schema_column_error("for row in data:\n    print(row['mean_temperature'])", schemas, prior)


def test_declared_context_files_are_inspected_and_missing_inputs_fail_closed(observations, tmp_path):
    context = {'available_files': [{'path': str(observations)}]}
    inspected = json.loads(_file_aware_context(context, 'Assess the data.', {}, {}))
    assert inspected['file_references'][0]['schema']['headers'] == HEADERS
    missing = tmp_path / 'missing.csv'
    with pytest.raises(ValueError, match='cannot be used'):
        _file_aware_context({'available_files': [str(missing)]}, 'Assess the data.', {}, {})


def test_load_then_validate_executes_with_verified_names(observations):
    original = observations.read_bytes()
    analysis = json.loads(valid_analysis())
    analysis['cell_plan'] = [
        dict(analysis['cell_plan'][0], title='Load records', purpose='Read records'),
        dict(analysis['cell_plan'][0], title='Validate records', purpose='Validate headers'),
    ]
    load_code = (
        'import csv\n'
        f'with open({observations.as_posix()!r}, encoding="utf-8-sig", newline="") as stream:\n'
        '    reader = csv.DictReader(stream)\n'
        '    headers = reader.fieldnames\n'
        '    data = list(reader)'
    )
    validation = f'required_columns = {HEADERS!r}\nassert set(required_columns) <= set(headers)\nassert float(data[0]["temperature_c"]) == 16.95'
    service = LLMCodeGenerator(config(), FakeClient(json.dumps(analysis), load_code, validation))
    cells = service.generate_cells(f'Read {observations.as_posix()}. Columns are date and temperature.')
    scope = {}
    for cell in cells:
        exec(cell['code'], scope)
    assert len(scope['data']) == 1
    assert observations.read_bytes() == original


def test_utf16_schema(tmp_path):
    path = tmp_path / 'utf16.csv'
    path.write_text('timestamp,temperature_c\n2026-01-01,23\n', encoding='utf-16')
    schema = inspect_table(str(path))
    assert schema['encoding'] == 'utf-16'
    assert schema['headers'] == ['timestamp', 'temperature_c']
