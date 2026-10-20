"""Oracle tests include deliberately wrong outputs: a checker must reject them."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location('ev_reference', Path(__file__).resolve().parents[1] / 'ev_reference.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'source.csv'
    rows = []
    for state, kind, make, year, value, ident in [
        ('WA', 'BEV', 'A', '2020', '0', '1'),
        ('WA', 'BEV', 'A', '2021', '100', '2'),
        ('WA', 'PHEV', 'B', '2021', 'NaN', '3'),
        ('WA', 'PHEV', 'B', '2022', '-1', '3'),
        ('OR', 'BEV', 'A', '2021', '500', '4'),
    ]:
        row = dict.fromkeys(ev.HEADERS, '')
        row.update({'State': state, 'Electric Vehicle Type': kind, 'Make': make,
                    'Model Year': year, 'Electric Range': value, 'DOL Vehicle ID': ident,
                    'VIN (1-10)': 'SAME_PREFIX', 'Base MSRP': '0'})
        rows.append(row)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=ev.HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_fixture(folder, expected):
    folder.mkdir()
    (folder / 'quality.json').write_text(json.dumps(expected['quality']))
    for filename, rows in expected['tables'].items():
        with (folder / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (folder / 'ev_summary.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"><title>Counts</title><rect/><text/><rect/></svg>')


def test_oracle_scope_missing_identifiers_and_ranges(source):
    result = ev.calculate(source)
    quality = result['quality']
    assert quality['rows_total'] == 5 and quality['rows_wa'] == 4
    assert quality['rows_other_or_missing_state'] == 1
    assert quality['duplicate_dol_ids'] == 1
    assert quality['missing_by_column']['County'] == 5
    assert quality['zero_electric_range_rows'] == 1
    assert quality['zero_base_msrp_rows'] == 4
    assert result['tables']['top_makes.csv'][0] == {'make': 'A', 'count': 2, 'percent': 50.0}
    bev, phev = result['tables']['range_by_type.csv']
    assert bev['mean_positive_range'] == 100 and bev['positive_range_count'] == 1
    assert bev['zero_or_missing_count'] == 1
    assert phev['invalid_range_count'] == 2 and phev['mean_positive_range'] == ''
    assert result['sqlite_crosscheck'] == 'passed'


def test_correct_exports_pass_without_executing_notebook(source, tmp_path):
    expected = ev.calculate(source)
    folder = tmp_path / 'exports'
    export_fixture(folder, expected)
    result = ev.check_generated(folder, expected)
    assert result['all_automated_checks_passed']
    assert result['chart_semantics_and_notebook_explanation_require_review']


@pytest.mark.parametrize('filename', ['quality.json', 'vehicle_types.csv', 'top_makes.csv', 'model_years.csv', 'range_by_type.csv'])
def test_corrupt_or_missing_export_fails(source, tmp_path, filename):
    expected = ev.calculate(source)
    folder = tmp_path / 'exports'
    export_fixture(folder, expected)
    (folder / filename).write_text('wrong output')
    assert ev.check_generated(folder, expected)['checks'][filename] is False


def test_numeric_error_fails_even_if_file_and_schema_exist(source, tmp_path):
    expected = ev.calculate(source)
    folder = tmp_path / 'exports'
    export_fixture(folder, expected)
    path = folder / 'range_by_type.csv'
    path.write_text(path.read_text().replace('100.0', '50.0'))
    assert ev.check_generated(folder, expected)['checks']['range_by_type.csv'] is False


def test_duplicate_output_keys_fail(source, tmp_path):
    expected = ev.calculate(source)
    folder = tmp_path / 'exports'
    export_fixture(folder, expected)
    path = folder / 'vehicle_types.csv'
    lines = path.read_text().splitlines()
    path.write_text('\n'.join(lines + [lines[1]]) + '\n')
    assert ev.check_generated(folder, expected)['checks']['vehicle_types.csv'] is False


def test_source_mutation_is_detected(source, tmp_path):
    expected = ev.calculate(source)
    folder = tmp_path / 'exports'
    export_fixture(folder, expected)
    source.write_bytes(source.read_bytes() + b'\n')
    assert ev.check_generated(folder, expected)['checks']['source_unchanged'] is False


def test_malformed_source_fails_instead_of_silently_skipping(source):
    with source.open('a') as f:
        f.write('too,few,columns\n')
    with pytest.raises(ValueError, match='malformed'):
        ev.calculate(source)


def test_blank_range_is_unknown_not_zero_and_infinity_is_invalid(source):
    text = source.read_text(encoding='utf-8-sig').replace(',100,', ',,').replace(',NaN,', ',inf,')
    source.write_text(text, encoding='utf-8-sig')
    result = ev.calculate(source)
    bev, phev = result['tables']['range_by_type.csv']
    assert bev['positive_range_count'] == 0 and bev['mean_positive_range'] == ''
    assert bev['zero_or_missing_count'] == 2
    assert result['quality']['zero_electric_range_rows'] == 1
    assert phev['invalid_range_count'] == 2


@pytest.mark.parametrize('execution_count,error,syntax,passed', [
    (1, False, 'x = 1', True),
    (None, False, 'x = 1', False),
    (1, True, 'x = 1', False),
    (1, False, 'x =', False),
])
def test_saved_notebook_execution_audit(tmp_path, execution_count, error, syntax, passed):
    path = tmp_path / 'generated.ipynb'
    path.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': [syntax],
        'execution_count': execution_count, 'outputs': [{'output_type': 'error'}] if error else []}]}))
    assert ev.check_notebook(path)['saved_execution_checks_passed'] is passed


def test_cli_never_rebaselines_changed_source(source, tmp_path):
    reference = tmp_path / 'reference.json'
    expected = ev.calculate(source)
    reference.write_text(json.dumps(expected))
    original = reference.read_bytes()
    source.write_bytes(source.read_bytes() + b'\n')
    result = subprocess.run([sys.executable, str(Path(ev.__file__)), '--source', str(source), '--reference', str(reference),
                             '--check-generated', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Do not silently rebaseline' in result.stderr
    assert reference.read_bytes() == original


def test_cli_reuses_frozen_reference_after_checkout_relocation(source, tmp_path):
    reference = tmp_path / 'reference.json'
    expected = ev.calculate(source)
    expected['source'] = 'C:/old-machine/no-longer-present/dataset.csv'
    reference.write_text(json.dumps(expected))
    frozen = reference.read_bytes()
    result = subprocess.run([sys.executable, str(Path(ev.__file__)), '--source', str(source),
                             '--reference', str(reference)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Reusing the frozen reference' in result.stdout
    assert reference.read_bytes() == frozen


def test_fractional_count_is_not_accepted_as_rounding(source, tmp_path):
    expected = ev.calculate(source)
    folder = tmp_path / 'exports'
    export_fixture(folder, expected)
    path = folder / 'vehicle_types.csv'
    path.write_text(path.read_text().replace(',2,', ',2.000000001,'))
    assert not ev.check_generated(folder, expected)['checks']['vehicle_types.csv']


def test_reverse_input_order_does_not_change_answers(source):
    expected = ev.calculate(source)
    with source.open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.reader(f))
    with source.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(records[0])
        writer.writerows(reversed(records[1:]))
    actual = ev.calculate(source)
    assert actual['quality'] == expected['quality']
    assert actual['tables'] == expected['tables']
