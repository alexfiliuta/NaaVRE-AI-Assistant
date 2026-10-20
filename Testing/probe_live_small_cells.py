"""Single-cell validation probes of a newly generated, explicitly reviewed notebook.

Runs only its second (data-quality) cell with supplied raw records. No model calls,
no repaired source, no file-writing generated cells. Expected failures are NOT
masked; exit 1 when any probe fails. A source review/hash gate is mandatory.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

import nbformat
from small_lab_reference import ROOT,sha


def energy(value='20',day='2026-01-01',identity='A'):
    return {'reading_date':day,'building_id':identity,'electricity_kwh':value}


def plant(value='8',day='2026-03-01',identity='X'):
    return {'measurement_date':day,'plant_id':identity,'treatment':'control',
            'height_cm':value,'leaf_count':'4','biomass_g':'2'}


def main(args):
    folder = args.case.resolve()
    assert (ROOT/'Testing/results/small-lab-tests').resolve() in folder.parents
    notebook = folder/'generated.ipynb'
    if sha(notebook)!=args.reviewed_sha256:
        raise ValueError('Notebook source does not match the explicitly reviewed hash')
    code = [c.source for c in nbformat.read(notebook,as_version=4).cells if c.cell_type=='code' and c.source.strip()][1]
    row = plant if args.kind=='plants' else energy
    cases = [('valid_zero',[row('0')],'valid'),('blank',[row('')],'unusable'),
        ('nan',[row('NaN')],'unusable'),('positive_infinity',[row('inf')],'unusable'),
        ('negative_infinity',[row('-inf')],'unusable'),('negative_measurement',[row('-1')],'unusable'),
        ('nonnumeric',[row('abc')],'unusable'),('invalid_date',[row(day='2026-02-30')],'unusable'),
        ('empty_identifier',[row(identity='')],'unusable'),('truncated_field',[row(None)],'unusable'),
        ('duplicate_observation',[row(),row()],'duplicate')]
    results = []
    began = time.monotonic()
    for name,rows,expectation in cases:
        namespace = {'raw_records':rows}
        result = {'case':name,'expectation':expectation,'status':'failed'}
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                exec(compile(code,str(notebook)+':quality-cell','exec'),namespace)
            valid = len(namespace['valid_records'])
            missing = len(namespace.get('missing_records',[]))
            rejected = len(namespace.get('rejected_records',namespace.get('validation_errors',[])))
            result.update(valid=valid,missing=missing,rejected=rejected)
            if expectation=='valid':
                assert valid==1 and missing==rejected==0
            elif expectation=='unusable':
                assert valid==0 and missing+rejected==1
            else:
                assert valid<=1 and rejected>=1, 'Duplicate key accepted without a validation diagnostic'
            result['status']='passed'
        except Exception as exc:
            result['error']=type(exc).__name__+': '+str(exc)
        result['stdout']=captured.getvalue()
        results.append(result)
    report = {'kind':args.kind,'reviewed_notebook_sha256':args.reviewed_sha256,
        'scope':'Only the unedited quality cell, with manual edge fixtures; not full notebook execution',
        'seconds':time.monotonic()-began,'passed':sum(r['status']=='passed' for r in results),
        'failed':sum(r['status']=='failed' for r in results),'cases':results}
    (folder/'live-cell-edge-probes.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return 1 if report['failed'] else 0


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--kind',choices=['plants','energy'],required=True)
    parser.add_argument('--reviewed-sha256',required=True)
    sys.exit(main(parser.parse_args()))
