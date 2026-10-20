"""Paired local-model pilot through the real agent and notebook generator.

Generation and execution are separate: review cells.json before --stage execute.
Never edits model cells or the source data. The execution guard is defense in depth,
not an operating-system sandbox; manual source review remains required.
"""
from __future__ import annotations
import argparse
import asyncio
import ast
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

import nbformat
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "naavre-ai-assistant"))
from naavre_ai_assistant.agent_runtime import AgentLimits, AgentRuntime
from naavre_ai_assistant.labs import LabRegistry
from naavre_ai_assistant.llm_service import LLMCodeGenerator, LLMConfig
from naavre_ai_assistant.mcp_gateway import MCPGateway

ROOT = Path(__file__).resolve().parents[1]
PROMPT = ('Compare electricity use in the two buildings over time. Check for missing readings '
          'and unusually high use, show a chart, and save the results. Use the lab data and '
          'protocol, and do not change the input file.')
TOOLS = ['list_files', 'inspect_dataset', 'search_knowledge', 'list_skills']

def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class MeasuredGenerator(LLMCodeGenerator):
    def __init__(self, config, folder):
        super().__init__(config)
        self.calls = []
        self.folder = folder

    def _request_content(self, prompt, **kwargs):
        row = {'kind': kwargs.get('response_kind', 'code'),
               'temperature': kwargs['temperature'], 'request_characters': len(prompt)}
        start = time.monotonic()
        print(f"MODEL REQUEST {len(self.calls)+1}: {row['kind']}", flush=True)
        try:
            content = super()._request_content(prompt, **kwargs)
            row.update(success=True, response_characters=len(content))
            response_dir=self.folder/'responses'
            response_dir.mkdir(exist_ok=True)
            save(response_dir/f'{len(self.calls)+1:03d}.json',
                 {'kind':row['kind'],'content':content})
            return content
        except Exception as exc:
            row.update(success=False, error=str(exc))
            raise
        finally:
            row['seconds'] = round(time.monotonic()-start, 3)
            self.calls.append(row)
            save(self.folder/'model-calls.json', self.calls)
            print(f"MODEL RESPONSE: {row['kind']} {row['seconds']} s", flush=True)

async def generate(args, folder):
    registry = LabRegistry()
    source = registry.get('energy').resolve('data', 'building_energy.csv')
    protocol = registry.get('energy').resolve('knowledge', 'energy-protocol.md')
    row = {'model': args.model, 'started_utc': datetime.now(timezone.utc).isoformat(),
           'prompt': PROMPT, 'input': str(source), 'input_sha256': digest(source),
           'protocol_sha256': digest(protocol), 'tools': TOOLS, 'context_window': 16384,
           'thinking': False, 'max_tokens': None, 'timeout_seconds': None,
           'execution_performed': False, 'status': 'starting', 'mode': args.stage,
        'source_hashes': {name: digest(ROOT/'naavre-ai-assistant'/'naavre_ai_assistant'/name)
                             for name in ['llm_service.py','agent_runtime.py','labs.py']}}
    with urlopen('http://127.0.0.1:11434/api/tags', timeout=15) as response:
        models = json.load(response)['models']
    row['model_details'] = next(x for x in models if x['name']==args.model)
    with urlopen('http://127.0.0.1:11434/api/version', timeout=15) as response:
        row['ollama'] = json.load(response)
    save(folder/'metrics.json', row)
    configuration = replace(LLMConfig.from_environment(), model=args.model,
        context_window=16384, thinking=False, max_tokens=None, timeout_seconds=None,
        agent_model=None, analysis_model=None, code_model=None)
    service = MeasuredGenerator(configuration, folder)
    started = time.monotonic()
    try:
        # Same warm-up for each model; record it separately from workflow latency.
        warm = time.monotonic()
        payload = json.dumps({'model': args.model,'stream': False,'think': False,
            'messages':[{'role':'user','content':'Reply with only OK.'}],
            'options':{'num_ctx':16384,'temperature':0,'num_predict':-1}}).encode()
        with urlopen(Request('http://127.0.0.1:11434/api/chat', data=payload,
                             headers={'Content-Type':'application/json'})) as response:
            warm_response = json.load(response)
        row['warmup_seconds'] = round(time.monotonic()-warm,3)
        row['warmup_metadata'] = {key: value for key,value in warm_response.items() if key!='message'}
        start = time.monotonic()
        manager = AgentRuntime(registry, folder/'state', service=service,
                               allowed_tools=TOOLS, limits=AgentLimits(max_steps=12))
        row['status'] = 'investigating'
        save(folder/'metrics.json',row)
        if args.stage=='generate-controlled':
            evidence=[]
            async with MCPGateway(registry.get('energy')) as gateway:
                await gateway.list_tools()
                for name,arguments in [('list_files',{'area':'data'}),
                    ('inspect_dataset',{'path':'building_energy.csv'}),
                    ('search_knowledge',{'query':'electricity missing threshold daily building'}),
                    ('list_skills',{})]:
                    result=await gateway.call_tool(name,arguments)
                    if result.get('isError'):
                        raise RuntimeError('Controlled MCP read failed: '+name)
                    evidence.append({'tool':name,'arguments':arguments,'result':result})
            context={'available_files':[source.as_posix()], 'verified_lab_evidence':evidence,
                     'lab_profile':registry.get('energy').public_dict()}
            row.update(investigation_status='fixed MCP evidence, no model-led investigation',
                investigation_seconds=round(time.monotonic()-start,3),tool_calls=4,reused_calls=0)
        else:
            run_id = manager.start('energy-model-comparison','energy',PROMPT)['run_id']
            await manager._runs[run_id].task
            snapshot = manager.get('energy-model-comparison',run_id)
            save(folder/'investigation.json',snapshot)
            row.update(investigation_status=snapshot['status'],
                       investigation_seconds=round(time.monotonic()-start,3),
                       tool_calls=sum(e['kind']=='observation' for e in snapshot['events']),
                       reused_calls=sum(e['kind']=='reuse' for e in snapshot['events']))
            if snapshot['status']!='completed':
                raise RuntimeError('Investigation did not complete: '+snapshot['status'])
            context = manager.verified_context('energy-model-comparison',run_id)
        save(folder/'context.json',context)
        row['context_sha256']=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
        # Only the per-run destination differs; scientific task/settings stay fixed.
        generation_prompt = (PROMPT + '\nUse focused Python cells. Input: ' + source.as_posix()
            + '. Save all derived files only under ' + (folder/'outputs').as_posix()
            + ' instead of the shared lab outputs, to keep this comparison isolated. '
            'Use standard-library CSV and SVG operations and display results inline.')
        if args.stage=='generate-controlled':
            generation_prompt += (' Keep reading raw CSV strings separate from date/numeric conversion. '
                'Use the verified lab protocol evidence supplied in context, not a guessed threshold.')
        row['generation_prompt'] = generation_prompt
        row['status']='analyzing'
        save(folder/'metrics.json',row)
        start=time.monotonic()
        analysis=await asyncio.to_thread(service.analyze_request,generation_prompt,context=context)
        row['analysis_seconds']=round(time.monotonic()-start,3)
        save(folder/'analysis.json',analysis)
        row['status']='generating'
        save(folder/'metrics.json',row)
        start=time.monotonic()
        cells=await asyncio.to_thread(service.generate_cells_from_analysis,generation_prompt,
                                      analysis=analysis,context=context)
        row['code_generation_seconds']=round(time.monotonic()-start,3)
        save(folder/'cells.json',cells)
        notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
            '# Building electricity comparison\n\nModel: **'+args.model+'**\n\n'+PROMPT
            +'\n\nSynthetic teaching data, not real building measurements. Unedited generated cells.')])
        for i,cell in enumerate(cells,1):
            notebook.cells += [nbformat.v4.new_markdown_cell(f"## {i}. {cell['title']}\n\n{cell['purpose']}"),
                               nbformat.v4.new_code_cell(cell['code'])]
        notebook.metadata['kernelspec']={'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'}
        nbformat.write(notebook,folder/'generated.ipynb')
        counts=[sum(bool(line.strip()) and not line.lstrip().startswith('#')
                    for line in c['code'].splitlines()) for c in cells]
        row.update(status='ready_for_source_review',cells=len(cells),
                   nonblank_noncomment_lines=sum(counts),lines_per_cell=counts,
                   max_cell_lines=max(counts),cell_titles=[c['title'] for c in cells],
                   generated_notebook=str(folder/'generated.ipynb'))
    except Exception as exc:
        row.update(failed_phase=row['status'],status='failed',error_type=type(exc).__name__,error=str(exc))
        logging.exception('Comparison generation failed')
    finally:
        row['generation_including_warmup_seconds']=round(time.monotonic()-started,3)
        row['workflow_seconds']=round(time.monotonic()-started-row.get('warmup_seconds',0),3)
        row['input_unchanged']=digest(source)==row['input_sha256']
        row['llm_calls']=len(service.calls)
        save(folder/'metrics.json',row)
        print(json.dumps(row,indent=2),flush=True)

def execute(args,folder):
    from nbclient import NotebookClient
    metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    if args.stage=='resnapshot':
        if metrics['status'] not in ('executed','executed_with_errors'):
            raise ValueError('Resnapshot is only for already reviewed/executed cells.')
        import shutil
        for name,target in [('executed.ipynb','execution-first.ipynb'),
                            ('metrics.json','execution-first-metrics.json')]:
            if not (folder/target).exists():
                shutil.copy2(folder/name,folder/target)
    elif metrics['status']!='ready_for_source_review':
        raise ValueError('Generation must complete before reviewed-source execution.')
    cells=json.loads((folder/'cells.json').read_text(encoding='utf-8'))
    allowed={'csv','pathlib','datetime','collections','statistics','math','xml','IPython',
             'json','html','io','re','itertools','typing','decimal','numbers','os'}
    for i,cell in enumerate(cells,1):
        tree=ast.parse(cell['code'])
        for node in ast.walk(tree):
            imports=[a.name for a in node.names] if isinstance(node,ast.Import) else (
                [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
            if any(name.split('.')[0] not in allowed for name in imports):
                raise ValueError(f'Cell {i} requires further review: imports {imports}')
            # An unused os import appears in the baseline. Permit that import,
            # not OS access: reviewed code must not actually load the os name.
            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id=='os':
                raise ValueError(f'Cell {i} requires further review: OS access')
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {
                'exec','eval','compile','__import__','breakpoint','input'}:
                raise ValueError(f'Cell {i} includes prohibited dynamic execution.')
    notebook=nbformat.read(folder/'generated.ipynb',as_version=4)
    output=(folder/'outputs').resolve()
    output.mkdir(exist_ok=True)
    # Added harness cells are clearly labelled and not included in model cell metrics.
    guard=f'''# EVALUATION HARNESS: not model-generated or scored
import sys as _evaluation_sys
from pathlib import Path as _EvaluationPath
_evaluation_output = _EvaluationPath({str(folder.resolve())!r})
def _evaluation_audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.remove', 'os.rename', 'os.rmdir', 'socket.connect'):
        raise PermissionError('Evaluation guard blocked: '+event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        mode, flags = args[1], args[2]
        writing = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (1|2|64|512|1024))
        if writing and not _EvaluationPath(args[0]).resolve().is_relative_to(_evaluation_output):
            raise PermissionError('Evaluation writes must stay inside the dedicated run folder')
    if event == 'os.mkdir' and not _EvaluationPath(args[0]).resolve().is_relative_to(_evaluation_output):
        raise PermissionError('Evaluation mkdir must stay inside the dedicated run folder')
_evaluation_sys.addaudithook(_evaluation_audit)
print('Evaluation guard active; model cells below remain unedited.')
'''
    notebook.cells.insert(1,nbformat.v4.new_code_cell(guard))
    notebook.cells.append(nbformat.v4.new_code_cell('''# EVALUATION HARNESS: export observable variables for independent checking
import json as _evaluation_json
import types as _evaluation_types
_evaluation_values = {k:v for k,v in list(globals().items()) if not k.startswith('_')
    and k not in ('In','Out','get_ipython','exit','quit','open')
    and not callable(v) and not isinstance(v,_evaluation_types.ModuleType)}
def _evaluation_serializable(value):
    if isinstance(value,dict):
        return {str(k):_evaluation_serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)):
        return [_evaluation_serializable(v) for v in value]
    return value
(_evaluation_output/'variables.json').write_text(_evaluation_json.dumps(_evaluation_serializable(_evaluation_values),
    default=str, indent=2),encoding='utf-8')
print('Independent-check snapshot saved.')
'''))
    start=time.monotonic()
    client=NotebookClient(notebook,timeout=120,allow_errors=True,kernel_name='python3',
                          resources={'metadata':{'path':str(folder.resolve())}},record_timing=True)
    client.execute()
    nbformat.write(notebook,folder/'executed.ipynb')
    generated=[c for c in notebook.cells if c.cell_type=='code' and not c.source.startswith('# EVALUATION HARNESS')]
    failures=[{'cell':i,'ename':o.get('ename'),'evalue':o.get('evalue')} for i,c in enumerate(generated,1)
              for o in c.outputs if o.output_type=='error']
    harness_failures=[{'ename':o.get('ename'),'evalue':o.get('evalue')}
        for c in notebook.cells if c.cell_type=='code' and c.source.startswith('# EVALUATION HARNESS')
        for o in c.outputs if o.output_type=='error']
    metrics.update(execution_performed=True,execution_seconds=round(time.monotonic()-start,3),
                   executed_cells=len(generated),error_cells=len({f['cell'] for f in failures}),
                   errors=failures,harness_errors=harness_failures,executed_notebook=str(folder/'executed.ipynb'),
                   status='executed_with_errors' if failures else 'executed',
                   model_code_unchanged=[c.source for c in generated]==[c['code'] for c in cells],
                   input_unchanged=digest(Path(metrics['input']))==metrics['input_sha256'])
    save(folder/'metrics.json',metrics)
    print(json.dumps(metrics,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['generate','generate-controlled','execute','resnapshot'],required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--folder',type=Path,required=True)
    args=parser.parse_args()
    folder=args.folder.resolve()
    folder.relative_to((ROOT/'Testing/results'/'energy-model-comparison').resolve())
    if args.stage in ('generate','generate-controlled'):
        folder.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout),
        logging.FileHandler(folder/(args.stage+'.log'),encoding='utf-8')])
    if args.stage in ('generate','generate-controlled'):
        asyncio.run(generate(args,folder))
    else:
        execute(args,folder)

if __name__=='__main__':
    main()
