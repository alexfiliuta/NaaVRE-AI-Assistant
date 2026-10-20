"""Exercise all four real MCP tools on the user-supplied large CSV, no LLM."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'naavre-ai-assistant'))
os.environ['NAAVRE_LABS_DIR'] = str(PROJECT / 'Labs')
os.environ['NAAVRE_ALLOWED_LABS'] = 'electric_vehicles'
from naavre_ai_assistant.labs import LabRegistry
from naavre_ai_assistant.mcp_gateway import MCPGateway
from naavre_ai_assistant.input_schema import inspect_table
from ev_reference import SOURCE, HEADERS, sha256


async def main():
    lab = LabRegistry().get('electric_vehicles')
    digest = sha256(SOURCE)
    records = []
    async with MCPGateway(lab) as gateway:
        tools = await gateway.list_tools()
        assert {tool['name'] for tool in tools} == {'list_files', 'inspect_dataset', 'search_knowledge', 'list_skills'}
        for name, arguments in [
            ('list_files', {'area': 'data'}),
            ('inspect_dataset', {'path': SOURCE.name}),
            ('search_knowledge', {'query': 'Washington population scope missing zero range outputs protocol'}),
            ('list_skills', {}),
        ]:
            start = time.perf_counter()
            result = await gateway.call_tool(name, arguments)
            assert not result.get('isError'), result
            payload = result['structuredContent']
            if name == 'inspect_dataset':
                assert payload['headers'] == HEADERS
                assert payload['sha256'] == digest
                assert payload['bytes'] == SOURCE.stat().st_size
                assert len(payload['preview']) == 5
                assert Path(payload['kernel_path']).resolve() == SOURCE.resolve()
            elif name == 'list_files':
                assert any(item['path'] == 'data/' + SOURCE.name for item in payload['files'])
            elif name == 'search_knowledge':
                assert payload['hits'] and all(hit['citation']['hash'] for hit in payload['hits'])
            else:
                assert payload['skills'][0]['id'] == 'ev-population-analysis'
            records.append({'tool': name, 'arguments': arguments,
                            'elapsed_seconds': time.perf_counter()-start, 'result': result})
    assert inspect_table(str(SOURCE))['headers'] == HEADERS
    assert sha256(SOURCE) == digest
    folder = PROJECT / 'Testing/results/electric-vehicles'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'mcp-verification.json').write_text(json.dumps({
        'status': 'passed', 'tools': records, 'source_sha256': digest,
        'llm_generation_tested': False, 'model_notebook_execution_tested': False,
    }, indent=2), encoding='utf-8')
    print('PASS: all four real stdio MCP tools and notebook schema inspection on the full EV file.')
    for item in records:
        print(f"  {item['tool']}: {item['elapsed_seconds']:.3f} s")


if __name__ == '__main__':
    asyncio.run(main())
