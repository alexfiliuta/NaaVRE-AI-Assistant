"""16 real MCP/runtime integration cases; SCRIPTED decisions, not Qwen quality tests."""
import asyncio
import itertools
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from small_lab_cases import ROOT, DATA, TOOLS, PROMPTS, digest, make_lab
from naavre_ai_assistant.agent_runtime import AgentLimits, AgentRuntime
from naavre_ai_assistant.labs import LabRegistry


MATRIX = list(itertools.product(['plants','energy'],[0,1],[0,1],[0,1]))


@pytest.mark.parametrize('kind,knowledge,skill,prompt_index',MATRIX,
    ids=[f'{kind}-k{k}-s{s}-p{p}' for kind,k,s,p in MATRIX])
def test_same_dataset_different_knowledge_skills_requests_propagate_without_leakage(
        tmp_path,monkeypatch,kind,knowledge,skill,prompt_index):
    lab = make_lab(tmp_path/'labs',kind,knowledge,skill)
    monkeypatch.setenv('NAAVRE_LABS_DIR',str(lab.parent))
    monkeypatch.setenv('NAAVRE_ALLOWED_LABS','case')
    source = lab/'data'/DATA[kind][0]
    original = digest(source)
    prompt = PROMPTS[kind][prompt_index]
    captured = []
    decisions = iter([
        {'action':'tool','tool':'inspect_dataset','arguments':{'path':DATA[kind][0]}},
        {'action':'tool','tool':'search_knowledge','arguments':{'query':'height electricity protocol readings treatment'}},
        {'action':'finish','summary':'Scripted test: inspect schema, protocol and curated skill before planning.'},
    ])

    async def model(value):
        captured.append(json.loads(value))
        return json.dumps(next(decisions))

    async def scenario():
        state = tmp_path/'agent-state'
        manager = AgentRuntime(LabRegistry(),state,model=model,allowed_tools=TOOLS,
            limits=AgentLimits(max_steps=12,run_seconds=45,operation_seconds=20))
        run_id = manager.start('context-matrix','case',prompt)['run_id']
        await asyncio.wait_for(manager._runs[run_id].task,50)
        snapshot = manager.get('context-matrix',run_id)
        assert snapshot['status']=='completed',snapshot['summary']
        observations = [e for e in snapshot['events'] if e['kind']=='observation']
        assert {e['tool'] for e in observations}==set(TOOLS)
        assert all(not e['result'].get('isError') for e in observations)
        schema = next(e['result']['structuredContent'] for e in observations if e['tool']=='inspect_dataset')
        headers = ['measurement_date','plant_id','treatment','height_cm','leaf_count','biomass_g'] if kind=='plants' else ['reading_date','building_id','electricity_kwh']
        assert schema['headers']==headers
        context = manager.verified_context('context-matrix',run_id)
        assert context['available_files']==[source.resolve().as_posix()]
        assert captured[0]['description']==prompt
        text = json.dumps(captured[-1])
        assert 'knowledge/protocol.md' in text
        assert digest(lab/'knowledge/protocol.md') in text
        assert 'Coverage-focused variant' in text if skill else 'Coverage-focused variant' not in text
        # No answers from the grader or cross-lab schemas are exposed to the model.
        assert 'small_lab_reference' not in text and 'PLANT_ENDPOINTS' not in text
        assert ('electricity_kwh' not in json.dumps(schema)) if kind=='plants' else ('plant_id' not in json.dumps(schema))
        if kind=='energy':
            assert ('30 kWh' if knowledge else '40 kWh') in text
        assert digest(source)==original==DATA[kind][1]
        record = {'kind':kind,'knowledge':knowledge,'skill':skill,'prompt':prompt,
            'status':'passed','decision_source':'scripted; NOT an LLM-quality experiment',
            'real_mcp_tools':sorted({e['tool'] for e in observations}),
            'input_sha256':original,'knowledge_sha256':digest(lab/'knowledge/protocol.md'),
            'skill_sha256':digest(lab/'skills/analysis-procedure.md'),'run_id':run_id,
            'state_dir':str(state),'source_unchanged':True}
        report = tmp_path/'context-matrix'
        report.mkdir(parents=True,exist_ok=True)
        (report/f'{kind}-k{knowledge}-s{skill}-p{prompt_index}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    asyncio.run(scenario())
