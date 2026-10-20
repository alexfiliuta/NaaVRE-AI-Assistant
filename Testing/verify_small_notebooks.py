"""Fresh-kernel execution and independent grading of explicitly reviewed notebooks.

Original cells are never edited. Extra cells, tagged test-harness, restrict writes
and capture values after execution. An audit hook is defence in depth, not an OS
sandbox: human source review plus a hash gate is required before each live case.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from portable_paths import reviewed_input_mapping, path_mapping_source

from small_lab_reference import (ROOT, NOTEBOOKS, SOURCES, REVIEWED_CODE_HASHES,
    reviewed_cells, sha, extract_original, grade_plants, grade_energy, check_svg,
    check_original_chart, grade_original_energy_exports,
    grade_plant_gains, grade_plant_groups, grade_plant_means)


def save(path,value):
    path.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')


def code_hash(notebook):
    sources = [c.source for c in notebook.cells if c.cell_type=='code' and c.source.strip()]
    return hashlib.sha256(json.dumps(sources,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()


def guard_cell(output):
    source = '''# TEST HARNESS: write boundary and network/process restrictions, NOT an OS sandbox.
import sys as _test_sys, os as _test_os
from pathlib import Path as _TestPath
_test_root = _TestPath(OUTPUT).resolve()
def _test_writable(value):
    if isinstance(value, int):
        return
    path = _TestPath(value).resolve()
    if path != _test_root and _test_root not in path.parents:
        raise PermissionError('Test notebook may write only under '+str(_test_root)+': '+str(path))
def _test_audit(event, args):
    if event == 'open':
        mode, flags = args[1], args[2]
        write = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (_test_os.O_WRONLY|_test_os.O_RDWR|_test_os.O_CREAT|_test_os.O_TRUNC|_test_os.O_APPEND))
        if write:
            _test_writable(args[0])
    elif event in {'os.mkdir','os.remove','os.rmdir'}:
        _test_writable(args[0])
    elif event in {'os.rename','os.replace'}:
        _test_writable(args[0]); _test_writable(args[1])
    elif event.startswith(('socket.','subprocess.')) or event in {'os.system','os.exec','os.spawn','os.fork'}:
        raise PermissionError('Network/process execution is forbidden in this notebook verification')
_test_sys.addaudithook(_test_audit)
print('Verification kernel:', _test_sys.executable)
'''.replace('OUTPUT',repr(str(output)))
    return nbformat.v4.new_code_cell(source,metadata={'tags':['test-harness']})


def execute(source,folder,kind,original=False,reviewed_sha=None):
    folder.mkdir(parents=True,exist_ok=True)
    output = folder/'outputs'
    output.mkdir(exist_ok=True)
    notebook = nbformat.read(source,as_version=4)
    before = code_hash(notebook)
    if original:
        reviewed_sources = reviewed_cells(kind)
        assert before==REVIEWED_CODE_HASHES[kind]
        inputs = [SOURCES[kind][0]]
        input_mapping = reviewed_input_mapping(reviewed_sources, inputs[0])
    else:
        if reviewed_sha is None or sha(source)!=reviewed_sha:
            raise ValueError('Live notebook needs its explicitly reviewed file SHA-256')
        inputs = list((folder/'labs/case/data').glob('*.csv'))
        assert len(inputs)==1
        input_mapping = {}
    input_before = {str(p):sha(p) for p in inputs}
    notebook.cells.insert(0,guard_cell(output))
    if input_mapping:
        notebook.cells.insert(0, nbformat.v4.new_code_cell(path_mapping_source(input_mapping),
            metadata={'tags': ['test-harness']}))
    if original:
        names = ['raw_records','plant_changes','treatment_summary','treatment_change_summary'] if kind=='plants' else ['raw_records','validation_report','building_summaries','flagged_records','overlap_comparison']
        capture = ('# TEST HARNESS: capture existing results, without correcting them.\n'
            'import json as _test_json\n'
            '_test_capture = {k:globals()[k] for k in '+repr(names)+'}\n'
            'with open('+repr(str(output/'harness-result.json'))+',"w",encoding="utf-8") as _test_file:\n'
            '    _test_json.dump(_test_capture,_test_file,indent=2,default=str)\n')
        notebook.cells.append(nbformat.v4.new_code_cell(capture,metadata={'tags':['test-harness']}))
    report = {'started_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),
        'kind':kind,'original_copy':original,'code_sha256':before,
        'model_cells':sum(c.cell_type=='code' and bool(c.source.strip()) and 'test-harness' not in c.metadata.get('tags',[]) for c in notebook.cells),
        'input_hashes_before':input_before,'execution_status':'running','numeric_status':'not_checked'}
    report['runtime_input_path_mapping'] = input_mapping
    start = time.monotonic()
    try:
        NotebookClient(notebook,timeout=600,kernel_name='python3',allow_errors=False,
            resources={'metadata':{'path':str(folder)}}).execute()
        report['execution_status']='passed'
    except Exception as exc:
        report.update(execution_status='failed',execution_error=str(exc))
    finally:
        report['execution_seconds']=time.monotonic()-start
        nbformat.write(notebook,folder/'executed.ipynb')
        unedited = [c.source for c in notebook.cells if c.cell_type=='code' and 'test-harness' not in c.metadata.get('tags',[]) and c.source.strip()]
        report['model_code_unchanged']=hashlib.sha256(json.dumps(unedited,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()==before
        report['input_hashes_after']={str(p):sha(p) for p in inputs}
        report['inputs_unchanged']=input_before==report['input_hashes_after']
    try:
        if report['execution_status']!='passed':
            raise AssertionError('Notebook execution failed; no passing numeric claim')
        if original:
            result = extract_original(kind,json.loads((output/'harness-result.json').read_text(encoding='utf-8')))
            tolerance = 5.1e-5 if kind=='plants' else 1e-8
        else:
            result = json.loads((output/'results.json').read_text(encoding='utf-8'))
            tolerance = 1e-8
        if kind=='plants':
            report['numeric_components']={}
            for name,grader in [('per_plant_endpoints_and_gains',grade_plant_gains),
                                ('treatment_date_height_summaries',grade_plant_groups),
                                ('mean_gain_by_treatment',grade_plant_means)]:
                try:
                    grader(result,tolerance)
                    report['numeric_components'][name]={'status':'passed'}
                except Exception as exc:
                    report['numeric_components'][name]={'status':'failed','error':str(exc)}
            grade_plants(result,tolerance)
        else:
            threshold = 30 if (folder/'metrics.json').exists() and json.loads((folder/'metrics.json').read_text())['knowledge']==1 else 40
            grade_energy(result,threshold,tolerance)
        report['numeric_status']='passed'
        if original and kind=='energy':
            report['csv_exports']=grade_original_energy_exports(output,result)
        save(folder/'normalized-results.json',result)
    except Exception as exc:
        report.update(numeric_status='failed',numeric_error=str(exc))
    charts = list(output.glob('*.svg'))
    try:
        assert len(charts)==1, 'Expected one exported SVG chart'
        report['chart_structure']=check_svg(charts[0])
        report['chart_structure']['status']='passed'
        if original:
            report['chart_geometry']=check_original_chart(kind,charts[0])
    except Exception as exc:
        report['chart_structure']={'status':'failed','error':str(exc)}
    report['all_checked_properties_passed']=(report['execution_status']==report['numeric_status']=='passed'
        and report['chart_structure']['status']=='passed' and report['inputs_unchanged'] and report['model_code_unchanged'])
    save(folder/'verification.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals',action='store_true')
    parser.add_argument('--case',type=Path)
    parser.add_argument('--kind',choices=['plants','energy'])
    parser.add_argument('--reviewed-sha256')
    parser.add_argument('--batch',default='originals-fresh-kernel')
    args = parser.parse_args()
    if args.originals:
        if not args.batch or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.batch):
            parser.error('Batch must be a simple identifier')
        batch = ROOT/'Testing/results/small-lab-tests'/args.batch
        batch.mkdir(parents=True,exist_ok=False)
        reports = [execute(NOTEBOOKS[k][0],batch/k,k,original=True) for k in ['plants','energy']]
    else:
        if not args.case or not args.kind or not args.reviewed_sha256:
            parser.error('Specify --originals OR --case, --kind and --reviewed-sha256')
        folder = args.case.resolve()
        safe_root = (ROOT/'Testing/results/small-lab-tests').resolve()
        if safe_root not in folder.parents:
            parser.error('Case must be inside the small-lab-tests result directory')
        if (folder/'executed.ipynb').exists():
            parser.error('Executed notebook already exists; preserve evidence, use a new case')
        reports = [execute(folder/'generated.ipynb',folder,args.kind,reviewed_sha=args.reviewed_sha256)]
    sys.exit(0 if all(r['all_checked_properties_passed'] for r in reports) else 1)
