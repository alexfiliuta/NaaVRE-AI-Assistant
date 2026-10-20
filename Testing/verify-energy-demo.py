"""Check the fixture and real MCP tools, without calling an LLM or notebook kernel."""
from __future__ import annotations

import asyncio
import csv
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from fixture_paths import LABS_ROOT

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'naavre-ai-assistant'))
os.environ['NAAVRE_LABS_DIR'] = str(LABS_ROOT)
os.environ['NAAVRE_ALLOWED_LABS'] = 'plants,energy'
from naavre_ai_assistant.labs import LabRegistry
from naavre_ai_assistant.mcp_gateway import MCPGateway


def payload(result):
    assert not result.get('isError'), result
    if 'structuredContent' in result:
        return result['structuredContent']
    return json.loads(next(item['text'] for item in result['content'] if item['type'] == 'text'))


async def main():
    source = LABS_ROOT / 'energy/data/building_energy.csv'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['reading_date', 'building_id', 'electricity_kwh']
        rows = list(reader)
    assert len(rows) == 28
    keys = [(row['reading_date'], row['building_id']) for row in rows]
    assert len(set(keys)) == 28
    assert {row['building_id'] for row in rows} == {'A', 'B'}
    assert len({date.fromisoformat(row['reading_date']) for row in rows}) == 14
    missing = [row for row in rows if not row['electricity_kwh']]
    assert [(row['reading_date'],row['building_id']) for row in missing] == [('2026-01-07','B')]
    valid = [row for row in rows if row['electricity_kwh']]
    assert all(math.isfinite(float(row['electricity_kwh'])) and float(row['electricity_kwh']) >= 0 for row in valid)
    flagged = [row for row in valid if float(row['electricity_kwh']) >= 40]
    assert [(row['reading_date'],row['building_id'],float(row['electricity_kwh'])) for row in flagged] == [('2026-01-10','B',80.0)]
    expected = {}
    for building in ('A','B'):
        readings = [float(row['electricity_kwh']) for row in valid if row['building_id'] == building]
        expected[building] = {'valid_readings':len(readings),'available_total_kwh':sum(readings),'mean_kwh':statistics.mean(readings)}
    assert expected['A']['available_total_kwh'] == 301
    assert expected['B']['available_total_kwh'] == 395
    common_dates = set(row['reading_date'] for row in valid if row['building_id']=='A') & set(row['reading_date'] for row in valid if row['building_id']=='B')
    assert len(common_dates) == 13
    common_totals = {building:sum(float(row['electricity_kwh']) for row in valid if row['building_id']==building and row['reading_date'] in common_dates) for building in ('A','B')}
    assert common_totals == {'A':276.0,'B':395.0}
    registry = LabRegistry()
    assert {'plants','energy'} <= {lab['id'] for lab in registry.list_labs()}
    lab = registry.get('energy')
    async with MCPGateway(lab) as gateway:
        tools = await gateway.list_tools()
        assert {tool['name'] for tool in tools} == {'list_files','inspect_dataset','search_knowledge','list_skills'}
        files = payload(await gateway.call_tool('list_files',{'area':'data'}))
        assert any(file['path']=='data/building_energy.csv' for file in files['files'])
        schema = payload(await gateway.call_tool('inspect_dataset',{'path':'building_energy.csv'}))
        assert schema['headers'] == ['reading_date','building_id','electricity_kwh']
        assert Path(schema['kernel_path']).resolve() == source.resolve()
        assert schema['sha256'] == source_hash and schema['sample_only'] is True
        knowledge = payload(await gateway.call_tool('search_knowledge',{'query':'electricity missing threshold daily building'}))
        assert knowledge['hits'] and any('synthetic' in hit['text'].lower() for hit in knowledge['hits'])
        skills = payload(await gateway.call_tool('list_skills',{}))
        assert skills['skills']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    report = {'fixture_checks':'passed','real_mcp_tools_checked':[tool['name'] for tool in tools],
              'input':source.as_posix(),'sha256':source_hash,'rows':len(rows),
              'missing_readings':missing,'review_flags':flagged,'expected_available_reading_summaries':expected,
              'common_dates':len(common_dates),'common_date_totals_kwh':common_totals,
              'llm_generation_tested':False,'notebook_execution_tested':False}
    output = PROJECT / 'Testing/results/energy-demo-authoring/verification.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    asyncio.run(main())
