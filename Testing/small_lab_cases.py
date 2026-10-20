"""Isolated small-lab test inputs and scientific contracts; no expected answers here."""
import hashlib
import json
from pathlib import Path

from fixture_paths import LABS_ROOT

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ['list_files', 'inspect_dataset', 'search_knowledge', 'list_skills']
DATA = {'plants': ('plant_growth.csv', 'c11c09cf5aa54d421e798ee6336d5109b28804a973b18c17418008d691a95c82'),
        'energy': ('building_energy.csv', '5c24bc29b79cbb9fae00e5bbf3c1f29f122988a3247027bd6ee6eae5857a1c5a')}
PROMPTS = {
    'plants': ['Compare the height gained by the plants in each treatment. Use the lab protocol and show a chart.',
               'How does plant height change over time for the two treatments? Compare their growth and show the individual plants.'],
    'energy': ['Compare electricity use in the two buildings over time, check quality, and flag high readings using the lab protocol.',
               'Which building uses more electricity on the dates they both have readings? Show the daily pattern and any readings needing review.'],
}
CONTRACTS = {
    'plants': '''Save results.json with keys rows (raw record count), gains (list of plant_id,treatment,first_date,last_date,first_height_cm,last_height_cm,height_gain_cm,n_measurements), summaries (list of treatment,measurement_date,n_plants,mean_height_cm,median_height_cm,min_height_cm,max_height_cm), and treatment_mean_gain (object mapping treatment to mean per-plant height gain). Sort endpoints chronologically, not by file order. Export full precision. Show an SVG chart and save it as chart.svg. Use descriptive results only.''',
    'energy': '''Save results.json with keys rows (raw record count), valid_count (available numeric readings), missing_count, rejected_count, summaries (object mapping building ID to available_readings,total_kwh,mean_kwh), flagged (list of reading_date,building_id,electricity_kwh), overlap (list of reading_date,building_A_kwh,building_B_kwh,difference_kwh), and common_totals (object mapping building ID to sum on overlap dates). Use the threshold supplied by the selected lab protocol. Retain flagged valid readings in summaries. Export full precision. Show a daily SVG chart with missing-data gaps and save it as chart.svg.''',
}
CASES = [
    {'id': 'plant-endpoints', 'lab': 'plants', 'knowledge': 0, 'skill': 0, 'prompt': 0},
    {'id': 'plant-timeseries', 'lab': 'plants', 'knowledge': 1, 'skill': 1, 'prompt': 1},
    {'id': 'energy-threshold40', 'lab': 'energy', 'knowledge': 0, 'skill': 0, 'prompt': 0},
    {'id': 'energy-threshold30', 'lab': 'energy', 'knowledge': 1, 'skill': 1, 'prompt': 1},
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_lab(root, kind, knowledge=0, skill=0, lab_id='case'):
    """Create a temporary lab with byte-identical source, not a new synthetic dataset."""
    source_name, expected_hash = DATA[kind]
    source = LABS_ROOT / kind / 'data' / source_name
    if digest(source) != expected_hash:
        raise ValueError('Original small dataset changed; review expected answers before testing.')
    lab = Path(root) / lab_id
    for area in ('data', 'knowledge', 'skills', 'outputs'):
        (lab / area).mkdir(parents=True)
    (lab / 'data' / source_name).write_bytes(source.read_bytes())
    original_protocol = next((LABS_ROOT / kind / 'knowledge').glob('*-protocol.md')).read_text(encoding='utf-8')
    if kind == 'energy' and knowledge:
        original_protocol = original_protocol.replace('40 kWh', '30 kWh')
    if kind == 'plants' and knowledge:
        original_protocol += '\nSupplement: order each plant by parsed date; compare both endpoints and treatment-date distributions. Never replace plant-level replication with row counts.'
    original_protocol += '\n\nRequired reproducible exports:\n' + CONTRACTS[kind]
    (lab / 'knowledge' / 'protocol.md').write_text(original_protocol, encoding='utf-8')
    original_skill = next((LABS_ROOT / kind / 'skills').glob('*.md')).read_text(encoding='utf-8')
    if skill:
        original_skill += '\nCoverage-focused variant: explicitly show denominators, excluded records and common-date coverage where relevant. Read the selected protocol before choosing analysis rules.'
    (lab / 'skills' / 'analysis-procedure.md').write_text(original_skill, encoding='utf-8')
    (lab / 'lab.json').write_text(json.dumps({
        'id': lab_id, 'name': kind + ' isolated test', 'description': 'Same fixed source; test knowledge/skill variants.',
        'style': 'State scientific limitations. Preserve source data.', 'frameworks': ['Python standard library', 'SVG'],
        'skills': ['analysis-procedure'], 'allowed_tools': TOOLS, 'enable_gbif': False,
    }), encoding='utf-8')
    return lab
