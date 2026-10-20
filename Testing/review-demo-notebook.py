"""Read-only collection of notebook source/output and independent CSV summaries."""
import csv
import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from fixture_paths import LABS_ROOT

root = Path(__file__).resolve().parents[1]
notebook_path = root / 'Testing/tests' / 'notebooks' / 'plants' / 'Success_Plant.ipynb'
source_path = LABS_ROOT / 'plants' / 'data' / 'plant_growth.csv'
report_path = root / 'Testing/results' / 'notebook-review' / 'saved-plant-review.json'
report_path.parent.mkdir(parents=True, exist_ok=True)

print(f'Notebook: {notebook_path}', flush=True)
print(f'Source: {source_path}', flush=True)
notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
cells = []
for index, cell in enumerate(notebook['cells']):
    item = {
        'index': index,
        'cell_type': cell['cell_type'],
        'source': ''.join(cell.get('source', [])),
        'execution_count': cell.get('execution_count'),
        'outputs': cell.get('outputs', []),
    }
    cells.append(item)
    print(f"Cell {index}: {item['cell_type']}; execution={item['execution_count']}; outputs={len(item['outputs'])}", flush=True)

with source_path.open(encoding='utf-8-sig', newline='') as handle:
    reader = csv.DictReader(handle)
    headers = reader.fieldnames
    rows = list(reader)
subjects = defaultdict(list)
groups = defaultdict(list)
for row in rows:
    parsed_date = date.fromisoformat(row['measurement_date'])
    height = float(row['height_cm'])
    subjects[row['plant_id']].append((parsed_date, height, row['treatment']))
    groups[(row['treatment'], parsed_date)].append((row['plant_id'], height))
gains = []
for plant, values in sorted(subjects.items()):
    values.sort()
    first, last = values[0], values[-1]
    gains.append({'plant_id': plant, 'treatment': first[2], 'first_date': str(first[0]), 'last_date': str(last[0]), 'first_height_cm': first[1], 'last_height_cm': last[1], 'height_gain_cm': last[1]-first[1], 'measurements': len(values)})
summaries = []
for (treatment, observation_date), values in sorted(groups.items()):
    heights = [height for _, height in values]
    summaries.append({'treatment': treatment, 'measurement_date': str(observation_date), 'n_plants': len({plant for plant, _ in values}), 'n_rows': len(values), 'mean_height_cm': statistics.mean(heights), 'median_height_cm': statistics.median(heights), 'min_height_cm': min(heights), 'max_height_cm': max(heights)})
duplicates = {str(key): count for key, count in Counter((row['plant_id'], row['measurement_date']) for row in rows).items() if count > 1}
artifacts = []
for output_root in (root/'outputs', notebook_path.parent/'outputs', LABS_ROOT/'plants'/'outputs'):
    if output_root.is_dir():
        for path in sorted(output_root.rglob('*.svg')):
            artifacts.append({'path': str(path), 'bytes': path.stat().st_size, 'text': path.read_text(encoding='utf-8')})
report = {'notebook_path': str(notebook_path), 'notebook_modified': notebook_path.stat().st_mtime, 'source_path': str(source_path), 'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(), 'headers': headers, 'rows': len(rows), 'plants': len(subjects), 'dates': sorted({row['measurement_date'] for row in rows}), 'duplicates': duplicates, 'expected_gains': gains, 'expected_treatment_summaries': summaries, 'cells': cells, 'svg_artifacts': artifacts}

checks = []
def check(name, condition):
    checks.append({'check': name, 'passed': bool(condition)})
    print(f"{'PASS' if condition else 'FAIL'}: {name}", flush=True)

def output_text(cell):
    return ''.join(''.join(output.get('text', [])) for output in cell['outputs'] if output['output_type'] == 'stream')

code_cells = [cell for cell in cells if cell['cell_type'] == 'code' and cell['source'].strip()]
check('Six generated cells have saved execution counts', len(code_cells) == 6 and [cell['execution_count'] for cell in code_cells] == list(range(1, 7)))
check('No saved notebook error outputs', not any(output['output_type'] == 'error' for cell in code_cells for output in cell['outputs']))
for cell in code_cells:
    compile(cell['source'], str(notebook_path), 'exec')
check('All cell sources compile without executing them', True)
check('Source CSV still matches the pre-build SHA256', report['source_sha256'] == 'c11c09cf5aa54d421e798ee6336d5109b28804a973b18c17418008d691a95c82')
check('Read and parse outputs retain all 24 source records', f'Read {len(rows)} raw records' in output_text(code_cells[0]) and f'Parsed {len(rows)} valid records.' in output_text(code_cells[2]) and 'No parsing errors detected.' in output_text(code_cells[2]))

observed_gains = {}
for line in output_text(code_cells[3]).splitlines():
    fields = line.split()
    if len(fields) == 8 and fields[0] in subjects:
        observed_gains[fields[0]] = fields
for gain in gains:
    fields = observed_gains.get(gain['plant_id'])
    matches = fields and fields[1] == gain['treatment'] and fields[2] == gain['first_date'] and fields[4] == gain['last_date'] and int(fields[7]) == gain['measurements']
    if matches:
        matches = all(math.isclose(float(fields[position]), round(gain[key], 1), abs_tol=1e-8) for position, key in ((3, 'first_height_cm'), (5, 'last_height_cm'), (6, 'height_gain_cm')))
    check(f"Printed endpoint values and height gain for {gain['plant_id']}", matches)
check('No plants were excluded in this complete dataset', 'No plants excluded.' in output_text(code_cells[3]))

observed_summaries = {}
for line in output_text(code_cells[4]).splitlines():
    fields = line.split()
    if len(fields) == 7 and re.fullmatch(r'\d{4}-\d{2}-\d{2}', fields[1]):
        observed_summaries[(fields[0], fields[1])] = fields
for summary in summaries:
    fields = observed_summaries.get((summary['treatment'], summary['measurement_date']))
    matches = fields and int(fields[2]) == summary['n_plants']
    if matches:
        matches = all(math.isclose(float(fields[position]), round(summary[key], 2), abs_tol=1e-8) for position, key in ((3, 'mean_height_cm'), (4, 'median_height_cm'), (5, 'min_height_cm'), (6, 'max_height_cm')))
    check(f"Printed treatment statistics for {summary['treatment']} on {summary['measurement_date']}", matches)
for treatment in sorted({gain['treatment'] for gain in gains}):
    treatment_gains = [gain['height_gain_cm'] for gain in gains if gain['treatment'] == treatment]
    expected_line = f'{treatment}: mean change = {statistics.mean(treatment_gains):.2f} cm (n={len(treatment_gains)} plants)'
    check(f'Printed mean height gain for {treatment}', expected_line in output_text(code_cells[4]))

saved_svgs = [output.get('data', {}).get('image/svg+xml')
              for cell in code_cells for output in cell['outputs']
              if output.get('data', {}).get('image/svg+xml')]
if not saved_svgs:
    raise SystemExit('The saved notebook has no inline SVG to verify.')
chart_source = saved_svgs[-1]
chart = ET.fromstring(''.join(chart_source) if isinstance(chart_source, list) else chart_source)
ns = {'svg': 'http://www.w3.org/2000/svg'}
polylines = chart.findall('svg:polyline', ns)
circles = chart.findall('svg:circle', ns)
axis_lines = chart.findall('svg:line', ns)
x_min, x_max = float(axis_lines[0].get('x1')), float(axis_lines[0].get('x2'))
y_bottom, y_top = float(axis_lines[0].get('y1')), float(axis_lines[1].get('y1'))
first_date = min(value[0] for values in subjects.values() for value in values)
last_date = max(value[0] for values in subjects.values() for value in values)
min_height = min(value[1] for values in subjects.values() for value in values)
max_height = max(value[1] for values in subjects.values() for value in values)
def expected_point(observation_date, height):
    x = x_min + (observation_date - first_date).days / (last_date - first_date).days * (x_max-x_min)
    y = y_bottom - (height-min_height)/(max_height-min_height)*(y_bottom-y_top)
    return (round(x, 6), round(y, 6))

expected_lines = Counter(tuple(expected_point(dt, height) for dt, height, _ in values) for values in subjects.values())
observed_lines = Counter(tuple(tuple(round(float(coordinate), 6) for coordinate in point.split(',')) for point in element.get('points').split()) for element in polylines)
expected_points = Counter(expected_point(dt, height) for values in subjects.values() for dt, height, _ in values)
observed_points = Counter((round(float(element.get('cx')), 6), round(float(element.get('cy')), 6)) for element in circles)
check('SVG contains six individual plant trajectories with correct data coordinates', len(polylines) == 6 and observed_lines == expected_lines)
check('SVG contains all 24 observation markers at their correct coordinates', len(circles) == 24 and observed_points == expected_points)
chart_labels = [element.text for element in chart.findall('svg:text', ns)]
check('SVG labels identify units and both treatments', all(label in chart_labels for label in ('Height (cm)', 'Measurement Date', 'control', 'fertilized')))
inline_svgs = [output.get('data', {}).get('image/svg+xml') for output in code_cells[5]['outputs'] if output.get('data', {}).get('image/svg+xml')]
check('Notebook saved an inline SVG display', bool(inline_svgs))
report['verification_checks'] = checks
report['all_checks_passed'] = all(item['passed'] for item in checks)
report['scope'] = 'Saved output and SVG geometry verification for this source dataset; generated cells were not re-executed.'
report['review_limitations'] = ['Chart uses individual plant trajectories, not the two treatment-mean lines described in the plan.', 'Generated code has no duplicate plant-date check and n_plants counts rows; this dataset has no duplicates.', 'Parser rejects an entire row when ancillary leaf_count or biomass_g is invalid, and does not reject non-finite numeric strings.']
report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('\nINDEPENDENT EXPECTED RESULTS', flush=True)
print(json.dumps({key: report[key] for key in ('rows', 'plants', 'dates', 'duplicates', 'expected_gains', 'expected_treatment_summaries')}, indent=2), flush=True)
print(f'\nEvidence report: {report_path}', flush=True)
if not report['all_checks_passed']:
    raise SystemExit(1)
