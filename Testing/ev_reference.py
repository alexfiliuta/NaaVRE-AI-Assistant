"""Independent, non-LLM EV oracle and output checker. Never sent to the model.

Uses streaming CSV/Counter, cross-checked against a separate SQLite aggregation.
Does NOT execute a supplied notebook or model-generated Python.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT / 'Labs/electric_vehicles/data/Electric Vehicle Population Data.csv'
RESULTS = PROJECT / 'Testing/results/electric-vehicles'
HEADERS = ['VIN (1-10)', 'County', 'City', 'State', 'Postal Code', 'Model Year',
           'Make', 'Model', 'Electric Vehicle Type',
           'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 'Electric Range',
           'Base MSRP', 'Legislative District', 'DOL Vehicle ID', 'Vehicle Location',
           'Electric Utility', '2020 Census Tract']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def number(text):
    try:
        value = float(text)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def calculate(source, *, sql_crosscheck=True):
    started = time.perf_counter()
    original_hash = sha256(source)
    missing = Counter({h: 0 for h in HEADERS})
    ids = set()
    types, makes, years = Counter(), Counter(), Counter()
    ranges = defaultdict(lambda: {'positive_range_count': 0, 'zero_or_missing_count': 0,
                                  'invalid_range_count': 0, 'positive_range_sum': 0.0})
    total = wa = duplicates = malformed = zero_range = zero_msrp = 0
    connection = sqlite3.connect(':memory:') if sql_crosscheck else None
    if connection:
        connection.execute('CREATE TABLE ev (kind TEXT, make TEXT, year TEXT, range REAL, status TEXT)')
    batch = []
    try:
        with Path(source).open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != HEADERS:
                raise ValueError('Source schema differs from the registered snapshot')
            for raw in reader:
                total += 1
                if None in raw or any(value is None for value in raw.values()):
                    malformed += 1
                    continue
                row = {k: v.strip() for k, v in raw.items()}
                missing.update(k for k in HEADERS if not row[k])
                ident = row['DOL Vehicle ID']
                if ident:
                    duplicates += ident in ids
                    ids.add(ident)
                if row['State'] != 'WA':
                    continue
                wa += 1
                kind, make, year = (row[k] or '(missing)' for k in
                                    ('Electric Vehicle Type', 'Make', 'Model Year'))
                types[kind] += 1
                makes[make] += 1
                years[year] += 1
                raw_range = row['Electric Range']
                value = number(raw_range)
                stats = ranges[kind]
                if raw_range == '' or value == 0:
                    stats['zero_or_missing_count'] += 1
                    status = 'unknown'
                elif value is None or value < 0:
                    stats['invalid_range_count'] += 1
                    status = 'invalid'
                else:
                    stats['positive_range_count'] += 1
                    stats['positive_range_sum'] += value
                    status = 'positive'
                zero_range += value == 0
                zero_msrp += number(row['Base MSRP']) == 0
                if connection:
                    batch.append((kind, make, year, value, status))
                    if len(batch) == 2000:
                        connection.executemany('INSERT INTO ev VALUES (?,?,?,?,?)', batch)
                        batch.clear()
        if connection:
            connection.executemany('INSERT INTO ev VALUES (?,?,?,?,?)', batch)
            for col, expected in [('kind', types), ('make', makes), ('year', years)]:
                actual = dict(connection.execute(f'SELECT {col}, COUNT(*) FROM ev GROUP BY {col}'))
                assert actual == dict(expected), f'SQLite disagreement: {col}'
            for kind, count, summed, mean in connection.execute(
                    "SELECT kind,COUNT(*),SUM(range),AVG(range) FROM ev WHERE status='positive' GROUP BY kind"):
                stats = ranges[kind]
                assert stats['positive_range_count'] == count
                assert math.isclose(stats['positive_range_sum'], summed, rel_tol=1e-12)
                assert math.isclose(summed/count, mean, rel_tol=1e-12)
        if sha256(source) != original_hash:
            raise ValueError('Input changed while generating reference')
        if malformed:
            raise ValueError(f'{malformed} malformed rows: clarify handling before benchmarking')
        def count_rows(counter, label, percent=False):
            return [{label: key, 'count': count, **({'percent': 100*count/wa} if percent else {})}
                    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]
        range_rows = []
        for kind, stats in sorted(ranges.items()):
            n = stats['positive_range_count']
            range_rows.append({'electric_vehicle_type': kind, **stats,
                               'mean_positive_range': stats['positive_range_sum']/n if n else ''})
        year_rows = [{'model_year': year, 'count': count} for year, count in
                     sorted(years.items(), key=lambda item: (not item[0].isdigit(),
                                                            int(item[0]) if item[0].isdigit() else item[0]))]
        return {'source': str(Path(source).resolve()), 'sha256': original_hash,
                'bytes': Path(source).stat().st_size,
                'quality': {'rows_total': total, 'rows_wa': wa,
                            'rows_other_or_missing_state': total-wa,
                            'duplicate_dol_ids': duplicates, 'malformed_rows': malformed,
                            'missing_by_column': dict(missing),
                            'zero_electric_range_rows': zero_range, 'zero_base_msrp_rows': zero_msrp},
                'tables': {'vehicle_types.csv': count_rows(types, 'electric_vehicle_type', True),
                           'top_makes.csv': count_rows(makes, 'make', True)[:10],
                           'model_years.csv': year_rows, 'range_by_type.csv': range_rows},
                'sqlite_crosscheck': 'passed' if connection else 'not_run',
                'reference_seconds': time.perf_counter()-started,
                'llm_generation_tested': False, 'model_notebook_execution_tested': False}
    finally:
        if connection:
            connection.close()


def check_generated(folder, expected):
    """Strict output contract, tolerant numeric formatting, no code execution."""
    folder = Path(folder)
    checks = {}
    try:
        actual_quality = json.loads((folder / 'quality.json').read_text(encoding='utf-8-sig'))
        checks['quality.json'] = all(actual_quality.get(k) == v for k, v in expected['quality'].items())
    except (OSError, ValueError, TypeError, AttributeError):
        checks['quality.json'] = False
    keys = {'vehicle_types.csv': 'electric_vehicle_type', 'top_makes.csv': 'make',
            'model_years.csv': 'model_year', 'range_by_type.csv': 'electric_vehicle_type'}
    for filename, rows in expected['tables'].items():
        try:
            with (folder / filename).open(encoding='utf-8-sig', newline='') as f:
                actual = list(csv.DictReader(f))
            key = keys[filename]
            by_key = {r[key]: r for r in actual}
            ok = len(by_key) == len(actual) == len(rows)
            for row in rows:
                other = by_key.get(str(row[key]), {})
                for col, val in row.items():
                    if isinstance(val, (float, int)):
                        parsed = number(other.get(col))
                        ok &= parsed is not None and (parsed == val if isinstance(val, int)
                                  else math.isclose(parsed, val, rel_tol=1e-9, abs_tol=1e-8))
                    else:
                        ok &= other.get(col) == str(val)
            if filename == 'top_makes.csv':
                ok &= [r.get(key) for r in actual] == [r[key] for r in rows]
            checks[filename] = bool(ok)
        except (OSError, ValueError, KeyError, TypeError):
            checks[filename] = False
    try:
        svg = ET.parse(folder / 'ev_summary.svg').getroot()
        checks['ev_summary.svg_structure_only'] = svg.tag.split('}')[-1] == 'svg' and len(list(svg)) > 3
    except (OSError, ET.ParseError):
        checks['ev_summary.svg_structure_only'] = False
    checks['source_unchanged'] = sha256(expected['source']) == expected['sha256']
    return {'checks': checks, 'all_automated_checks_passed': all(checks.values()),
            'chart_semantics_and_notebook_explanation_require_review': True,
            'note': 'A failure may be a filename/schema contract mismatch, not necessarily arithmetic. Inspect it.'}


def check_notebook(path):
    """Audit saved execution evidence; do not execute or trust notebook code."""
    notebook = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    cells = [c for c in notebook.get('cells', []) if c.get('cell_type') == 'code'
             and ''.join(c.get('source', [])).strip()]
    details = []
    for index, cell in enumerate(cells, 1):
        source = ''.join(cell.get('source', []))
        try:
            ast.parse(source)
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False
        details.append({'code_cell': index, 'python_syntax_valid': syntax_ok,
                        'recorded_as_executed': cell.get('execution_count') is not None,
                        'no_recorded_errors': not any(o.get('output_type') == 'error'
                                                     for o in cell.get('outputs', []))})
    return {'nonempty_code_cells': len(cells), 'cells': details,
            'saved_execution_checks_passed': bool(cells) and all(
                all(v for k, v in c.items() if k != 'code_cell') for c in details),
            'notebook_sha256': sha256(path),
            'limitation': 'Saved execution counts are evidence, not a fresh rerun or proof of output provenance. '
                          'IPython magic syntax is not accepted by this plain-Python audit.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--check-generated', type=Path)
    parser.add_argument('--notebook', type=Path, help='Audit saved notebook, without executing its code')
    parser.add_argument('--reference', type=Path, default=RESULTS / 'reference.json')
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    reference = args.reference
    if args.check_generated:
        if not reference.exists():
            parser.error('Create the reference before generating the notebook; reference is missing.')
        expected = json.loads(reference.read_text(encoding='utf-8'))
        # A frozen reference records the original location for provenance. The
        # same data can now live in a relocated checkout; identity is its hash.
        if sha256(args.source) != expected['sha256']:
            parser.error('The input changed after the reference was created. Do not silently rebaseline.')
    else:
        if args.notebook:
            parser.error('--notebook requires --check-generated')
        if reference.exists():
            expected = json.loads(reference.read_text(encoding='utf-8'))
            if expected['sha256'] != sha256(args.source):
                parser.error('Reference already exists for another snapshot. Use a new --reference path.')
            print('Reusing the frozen reference; no answers overwritten.')
        else:
            expected = calculate(args.source)
            reference.parent.mkdir(parents=True, exist_ok=True)
            with reference.open('x', encoding='utf-8') as f:
                json.dump(expected, f, indent=2)
    print(json.dumps({k: v for k, v in expected.items() if k != 'tables'}, indent=2))
    print(json.dumps(expected['tables'], indent=2))
    if args.check_generated:
        report = check_generated(args.check_generated, expected)
        report['reference_sha256'] = sha256(reference)
        report['generated_outputs'] = str(args.check_generated.resolve())
        if args.notebook:
            report['notebook_audit'] = check_notebook(args.notebook)
            report['all_automated_checks_passed'] &= report['notebook_audit']['saved_execution_checks_passed']
        report['checked_at'] = datetime.now(timezone.utc).isoformat()
        output = RESULTS / ('check-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        if not report['all_automated_checks_passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
