"""Independent, fixed answers for the two SMALL fixtures. Never supplied to Qwen.

No production analysis functions are imported. Source hashes pin the fixtures;
literal answers below are independently checked by tests. This is not an oracle
for arbitrary scientific data or a proof that a language model always works.
"""
import csv
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from fixture_paths import LABS_ROOT

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'plants': (LABS_ROOT/'plants/data/plant_growth.csv',
               'c11c09cf5aa54d421e798ee6336d5109b28804a973b18c17418008d691a95c82'),
    'energy': (LABS_ROOT/'energy/data/building_energy.csv',
               '5c24bc29b79cbb9fae00e5bbf3c1f29f122988a3247027bd6ee6eae5857a1c5a'),
}
NOTEBOOKS = {
    'plants': (ROOT/'Testing/tests/notebooks/plants/Success_Plant.ipynb',
               '286784695e37c044158fa087f63f0db5854648fc5aa6a94348b2605c5bfadf5b'),
    'energy': (ROOT/'Testing/tests/notebooks/energy/Energy_Test.ipynb',
               'a3342010486d29d938d033964deba49f23c7cf9dae79488e1310eb2cf2eba6bc'),
}
REVIEWED_CODE_HASHES = {
    'plants':'6ac58e2a36bba0c3c44ebc5fc8ae869f17017cda26de8467d10682bb9a0b58cb',
    'energy':'583a939241d0d82d416001c7421b18b62146a41b76bd8142df17eb17bb2559cc',
}
PLANT_ENDPOINTS = {
    'P01': ('control', 8.2, 13.8, 5.6), 'P02': ('control', 8.5, 14.1, 5.6),
    'P03': ('control', 7.9, 13.5, 5.6), 'P04': ('fertilized', 8.4, 18.3, 9.9),
    'P05': ('fertilized', 8.1, 17.8, 9.7), 'P06': ('fertilized', 8.6, 18.7, 10.1),
}
# treatment, date -> mean, median, min, max; three DISTINCT plants per group.
PLANT_GROUPS = {
    ('control','2026-03-01'): (8.2,8.2,7.9,8.5),
    ('fertilized','2026-03-01'): (25.1/3,8.4,8.1,8.6),
    ('control','2026-03-08'): (10.1,10.1,9.8,10.4),
    ('fertilized','2026-03-08'): (11.2,11.2,10.9,11.5),
    ('control','2026-03-15'): (35.9/3,12.0,11.7,12.2),
    ('fertilized','2026-03-15'): (44.0/3,14.7,14.3,15.0),
    ('control','2026-03-22'): (13.8,13.8,13.5,14.1),
    ('fertilized','2026-03-22'): (54.8/3,18.3,17.8,18.7),
}
ENERGY_VALUES = {
    'A': [20,21,15,14,23,24,25,26,27,16,15,24,25,26],
    'B': [25,26,18,17,28,29,None,31,32,80,19,29,30,31],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(kind):
    path, expected = SOURCES[kind]
    if sha(path) != expected:
        raise ValueError('Fixture changed; do not apply stale reference answers: '+str(path))
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def reviewed_cells(kind):
    # Frozen unedited generated source makes the unit tests reproducible even when
    # a scientist later renames/edits their own notebook. Provenance is in the fixture.
    path = ROOT/'Testing/tests/fixtures/reviewed-small-lab-cells.json'
    fixture = json.loads(path.read_text(encoding='utf-8'))
    sources = fixture[kind]['code']
    # Jupyter may autosave outputs/metadata: executable source is what was reviewed.
    actual = hashlib.sha256(json.dumps(sources,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
    if actual != REVIEWED_CODE_HASHES[kind]:
        raise ValueError('Code changed; inspect source and update reviewed hash before executing: '+str(path))
    return sources


def run_calculation_cell(kind,index,namespace):
    """One cell = one unit under test; caller supplies its dependencies explicitly."""
    cells = reviewed_cells(kind)
    exec(compile(cells[index], f'{NOTEBOOKS[kind][0]}:cell-{index+1}', 'exec'), namespace)
    return namespace


def calculation_cells(kind, rows, indices=None):
    """Execute ONLY already-reviewed pure calculation cells, not file I/O/plots.

    This runs the original generated implementation, not a rewritten solution.
    The hash gate is review protection, NOT an OS security sandbox.
    """
    namespace = {'raw_records': [dict(row) for row in rows]}
    indices = indices if indices is not None else ((2,3,4) if kind=='plants' else (1,2,3))
    for i in indices:
        run_calculation_cell(kind,i,namespace)
    return namespace


def canonical_date(value):
    # Accept ISO date or ISO datetime exports, but do not accept arbitrary labels.
    text = str(value)
    if len(text)==10:
        return date.fromisoformat(text).isoformat()
    return datetime.fromisoformat(text.replace('Z','+00:00')).date().isoformat()


def number(actual, expected, label, tolerance=1e-8):
    if isinstance(actual, bool) or not isinstance(actual, (int,float,Decimal)):
        raise AssertionError(f'{label}: not a numeric value: {actual!r}')
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=tolerance):
        raise AssertionError(f'{label}: got {actual!r}; expected {expected!r}')


def unique_index(rows, key):
    result = {}
    for row in rows:
        value = key(row)
        if value in result:
            raise AssertionError('Duplicate result key: '+str(value))
        result[value] = row
    return result


def energy_expected(threshold=40):
    summaries = {}
    flags = []
    for building, values in ENERGY_VALUES.items():
        valid = [Decimal(str(x)) for x in values if x is not None]
        summaries[building] = {'available_readings': len(valid), 'total_kwh': float(sum(valid)),
                               'mean_kwh': float(sum(valid)/len(valid))}
        flags.extend({'reading_date': f'2026-01-{i+1:02}', 'building_id': building, 'electricity_kwh': x}
                     for i,x in enumerate(values) if x is not None and x>=threshold)
    overlap = [{'reading_date':f'2026-01-{i+1:02}', 'building_A_kwh':a, 'building_B_kwh':b,
                'difference_kwh':a-b} for i,(a,b) in enumerate(zip(ENERGY_VALUES['A'],ENERGY_VALUES['B']))
               if a is not None and b is not None]
    return {'rows':28, 'valid_count':27, 'missing_count':1, 'rejected_count':0,
            'summaries':summaries,'flagged':flags,'overlap':overlap,
            'common_totals':{'A':276,'B':395}}


def grade_plants(result, tolerance=1e-8):
    number(result['rows'],24,'rows')
    grade_plant_gains(result,tolerance)
    grade_plant_groups(result,tolerance)
    grade_plant_means(result,tolerance)


def grade_plant_gains(result,tolerance=1e-8):
    gains = unique_index(result['gains'], lambda r:r['plant_id'])
    assert set(gains)==set(PLANT_ENDPOINTS), 'Missing/extra plants'
    for pid,(treatment,first,last,gain) in PLANT_ENDPOINTS.items():
        row = gains[pid]
        assert row['treatment']==treatment
        assert canonical_date(row['first_date'])=='2026-03-01'
        assert canonical_date(row['last_date'])=='2026-03-22'
        for field,expected in [('first_height_cm',first),('last_height_cm',last),('height_gain_cm',gain),('n_measurements',4)]:
            number(row[field],expected,pid+'/'+field,tolerance)


def grade_plant_groups(result,tolerance=1e-8):
    summaries = unique_index(result['summaries'],lambda r:(r['treatment'],canonical_date(r['measurement_date'])))
    assert set(summaries)==set(PLANT_GROUPS), 'Missing/extra treatment-date groups'
    for key,values in PLANT_GROUPS.items():
        number(summaries[key]['n_plants'],3,str(key)+'/n_plants')
        for field,value in zip(['mean_height_cm','median_height_cm','min_height_cm','max_height_cm'],values):
            number(summaries[key][field],value,str(key)+'/'+field,tolerance)


def grade_plant_means(result,tolerance=1e-8):
    assert set(result['treatment_mean_gain'])=={'control','fertilized'}
    for treatment,value in [('control',5.6),('fertilized',9.9)]:
        number(result['treatment_mean_gain'][treatment],value,treatment+'/mean_gain',tolerance)


def grade_energy(result, threshold=40, tolerance=1e-8):
    expected = energy_expected(threshold)
    for key in ['rows','valid_count','missing_count','rejected_count']:
        number(result[key],expected[key],key)
    for key in ['summaries','common_totals']:
        assert set(result[key])=={'A','B'}, key+' must describe A and B'
    for building,summary in expected['summaries'].items():
        for key,value in summary.items():
            number(result['summaries'][building][key],value,building+'/'+key,tolerance)
        number(result['common_totals'][building],expected['common_totals'][building],building+'/common_totals',tolerance)
    for collection,key_fields,value_fields in [
        ('flagged',('reading_date','building_id'),('electricity_kwh',)),
        ('overlap',('reading_date',),('building_A_kwh','building_B_kwh','difference_kwh'))]:
        key = lambda r:tuple(canonical_date(r[f]) if f=='reading_date' else r[f] for f in key_fields)
        actual_index = unique_index(result[collection],key)
        expected_index = unique_index(expected[collection],key)
        assert set(actual_index)==set(expected_index), collection+' has missing/extra keys'
        for row_key,row in expected_index.items():
            for field in value_fields:
                number(actual_index[row_key][field],row[field],collection+'/'+str(row_key)+'/'+field,tolerance)


def extract_original(kind, namespace):
    """An explicit variable-name adapter, not a numerical correction."""
    if kind=='plants':
        return {'rows':len(namespace['raw_records']),
            'gains':[{**r,'height_gain_cm':r['change_cm']} for r in namespace['plant_changes']],
            'summaries':[{**r,'measurement_date':r['date']} for r in namespace['treatment_summary']],
            'treatment_mean_gain':{t:sum(v)/len(v) for t,v in namespace['treatment_change_summary'].items()}}
    overlap = namespace['overlap_comparison']
    return {'rows':len(namespace['raw_records']),**namespace['validation_report'],
        'summaries':namespace['building_summaries'],'flagged':namespace['flagged_records'],
        'overlap':overlap,'common_totals':{b:sum(r[f'building_{b}_kwh'] for r in overlap) for b in ['A','B']}}


def check_svg(path):
    """Structural chart check only; numeric answers are checked separately."""
    svg = ET.parse(path).getroot()
    assert svg.tag=='{http://www.w3.org/2000/svg}svg', 'Not a namespaced SVG'
    assert any(e.tag.split('}')[-1] in {'polyline','line','path','circle','rect'} for e in svg.iter()), 'No plotted shapes'
    assert any(e.tag.split('}')[-1]=='text' and ''.join(e.itertext()).strip() for e in svg.iter()), 'No chart text/labels'
    return {'check':'SVG structure and labels only; not full scientific chart validation',
            'bytes':Path(path).stat().st_size}


def check_original_chart(kind,path):
    """Verify every plotted observation/segment for the TWO reviewed renderers.

    Their reviewed layout determines coordinates, not their computed variables.
    CSV/reference values determine expected heights/energy. Not a generic SVG grader.
    """
    root = ET.parse(path).getroot()
    ns = {'s':'http://www.w3.org/2000/svg'}
    coord = lambda value:round(float(value),6)
    if kind=='plants':
        rows = read_rows('plants')
        expected_lines = []
        expected_points = []
        for pid in PLANT_ENDPOINTS:
            records = sorted([r for r in rows if r['plant_id']==pid],key=lambda r:r['measurement_date'])
            color = '#1f77b4' if records[0]['treatment']=='control' else '#ff7f0e'
            points = tuple((coord(80+560*i/3),coord(420-(float(r['height_cm'])-7.9)/(18.7-7.9)*360)) for i,r in enumerate(records))
            expected_lines.append((color,points))
            expected_points.extend((color,x,y) for x,y in points)
        actual_lines = [(e.get('stroke'),tuple(tuple(coord(v) for v in p.split(',')) for p in e.get('points').split())) for e in root.findall('s:polyline',ns)]
        actual_points = [(e.get('fill'),coord(e.get('cx')),coord(e.get('cy'))) for e in root.findall('s:circle',ns)]
        assert Counter(actual_lines)==Counter(expected_lines), 'Plant trajectory coordinates incorrect'
        assert Counter(actual_points)==Counter(expected_points), 'Plant marker coordinates incorrect'
        return {'status':'passed','scope':'All 24 plant markers and all six four-point trajectories match source data','observations':24,'trajectories':6}
    colors = {'A':'#1f77b4','B':'#d62728'}
    position = lambda i,value:(coord(70+690*i/13),coord(420-(value-7.4)/79.2*360))
    expected_points,expected_lines = [],[]
    for building,values in ENERGY_VALUES.items():
        for i,value in enumerate(values):
            if value is not None:
                expected_points.append((colors[building],*position(i,value)))
            if i and value is not None and values[i-1] is not None:
                expected_lines.append((colors[building],*position(i-1,values[i-1]),*position(i,value)))
    actual_points = []
    for e in root.findall('s:circle',ns):
        if e.get('r') not in {'4','6'}:
            continue  # Inner high-reading dot is not an extra observation.
        color = e.get('fill') if e.get('fill') in colors.values() else e.get('stroke')
        if color not in colors.values():
            continue
        x,y = coord(e.get('cx')),coord(e.get('cy'))
        if x==660 and y in {85,105}:
            continue  # Explicit legend positions in the reviewed renderer.
        actual_points.append((color,x,y))
    actual_lines = []
    for e in root.findall('s:line',ns):
        color = e.get('stroke')
        if color not in colors.values():
            continue
        x1,y1,x2,y2 = [coord(e.get(k)) for k in ['x1','y1','x2','y2']]
        if (x1,x2)==(650,670) and y1==y2 and y1 in {85,105}:
            continue
        actual_lines.append((color,x1,y1,x2,y2))
    assert Counter(actual_points)==Counter(expected_points), 'Energy observation coordinates incorrect'
    assert Counter(actual_lines)==Counter(expected_lines), 'Energy segments incorrect (including missing-day gap)'
    flagged = [e for e in root.findall('s:circle',ns) if e.get('r')=='6' and e.get('stroke')==colors['B']]
    assert len(flagged)==1 and (coord(flagged[0].get('cx')),coord(flagged[0].get('cy')))==position(9,80)
    return {'status':'passed','scope':'All 27 readings, all 24 connecting segments, missing-day gap and high-reading marker match source data',
            'observations':27,'segments':24,'missing_gap_checked':True}


def grade_original_energy_exports(output,base):
    def csv_rows(name):
        with (Path(output)/name).open(encoding='utf-8-sig',newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in list(row):
                if key in {'available_readings','total_kwh','mean_kwh','electricity_kwh','building_A_kwh','building_B_kwh','difference_kwh'}:
                    row[key]=float(row[key])
        return rows
    result = {**base,
        'summaries':unique_index(csv_rows('energy_summary.csv'),lambda r:r['building_id']),
        'overlap':csv_rows('energy_overlap.csv'),'flagged':csv_rows('energy_flagged.csv')}
    grade_energy(result)
    return {'status':'passed','scope':'Every row and numeric field in all three exported CSVs'}


def check_new_plant_gain_chart(path):
    """Numeric bar geometry for the explicitly reviewed plant-endpoints renderer."""
    root = ET.parse(path).getroot()
    bars = [e for e in root.findall('{http://www.w3.org/2000/svg}rect') if e.get('stroke-width')=='0.5']
    assert len(bars)==2
    bars.sort(key=lambda e:float(e.get('x')))
    for bar,x,gain in zip(bars,[128,368],[5.6,9.9]):
        for attribute,expected in [('x',x),('y',320-gain/9.9*260),('width',144),('height',gain/9.9*260)]:
            number(float(bar.get(attribute)),expected,'gain bar '+attribute,tolerance=1e-6)
    return {'status':'passed','scope':'Both treatment mean-gain bar geometries match independent 5.6/9.9 cm answers'}


def check_new_energy_chart(path):
    """Every data point/segment/flag in the reviewed threshold30 renderer."""
    root = ET.parse(path).getroot()
    ns = {'s':'http://www.w3.org/2000/svg'}
    coord = lambda value:round(float(value),6)
    position = lambda i,value:(coord(80+680*i/13),coord(420-(value-7.4)/79.2*360))
    colors = {'A':'#2196F3','B':'#FF5722'}
    expected_points,expected_lines,expected_flags = [],[],[]
    for building,values in ENERGY_VALUES.items():
        segment = []
        for i,value in enumerate(values):
            if value is None:
                if segment:
                    expected_lines.append((colors[building],tuple(segment)))
                    segment=[]
                continue
            x,y = position(i,value)
            segment.append((x,y))
            expected_points.append((colors[building],x,y))
            if value>=30:
                expected_flags.append((x,y))
        if segment:
            expected_lines.append((colors[building],tuple(segment)))
    points = [(e.get('fill'),coord(e.get('cx')),coord(e.get('cy'))) for e in root.findall('s:circle',ns)]
    lines = [(e.get('stroke'),tuple(tuple(coord(v) for v in p.split(',')) for p in e.get('points').split())) for e in root.findall('s:polyline',ns)]
    flags = []
    for e in root.findall('s:polygon',ns):
        if e.get('stroke-width')!='1.5':
            continue  # The separate legend diamond is not a data point.
        corners = [tuple(float(v) for v in p.split(',')) for p in e.get('points').split()]
        assert len(corners)==4
        flags.append((coord(sum(p[0] for p in corners)/4),coord(sum(p[1] for p in corners)/4)))
    assert Counter(points)==Counter(expected_points), 'New energy marker coordinates incorrect'
    assert Counter(lines)==Counter(expected_lines), 'New energy series/gap coordinates incorrect'
    assert Counter(flags)==Counter(expected_flags), 'New energy review-marker positions incorrect'
    return {'status':'passed','scope':'All 27 observations, three polyline segments including the missing-day break, and five >=30 kWh review markers match source values',
            'observations':27,'polyline_segments':3,'flags':5,'missing_gap_checked':True}
