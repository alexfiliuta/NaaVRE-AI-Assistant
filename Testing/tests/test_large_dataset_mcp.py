"""Large source support must not increase evidence size or weaken confinement."""
import hashlib
import json

import pytest

from test_labs import lab_registry
from naavre_ai_assistant.labs import LabError, _table_snapshot
from naavre_ai_assistant.mcp_gateway import MCPGateway
from naavre_ai_assistant.input_schema import inspect_table


def test_snapshot_hashes_full_file_but_retains_only_prefix(tmp_path):
    path = tmp_path / 'large.csv'
    data = b'a,b\n' + b'1,2\n' * (2 * 1024 * 1024 + 1)
    path.write_bytes(data)
    prefix, digest, size = _table_snapshot(path)
    assert len(data) > 8 * 1024 * 1024
    assert prefix == data[:65_536]
    assert size == len(data)
    assert digest == hashlib.sha256(data).hexdigest()
    assert inspect_table(str(path))['headers'] == ['a', 'b']
    with pytest.raises(LabError, match='inspection limit'):
        _table_snapshot(path, max_bytes=1024)


@pytest.mark.asyncio
async def test_real_mcp_large_csv_has_bounded_evidence(lab_registry):
    path = lab_registry.get('a').root / 'data' / 'large.csv'
    data = b'a,b\n' + b'1,2\n' * (2 * 1024 * 1024 + 1)
    path.write_bytes(data)
    async with MCPGateway(lab_registry.get('a')) as gateway:
        result = await gateway.call_tool('inspect_dataset', {'path': 'large.csv'})
    assert not result.get('isError'), result
    schema = result['structuredContent']
    assert schema['headers'] == ['a', 'b']
    assert schema['preview'] == [['1', '2']] * 5
    assert schema['sha256'] == hashlib.sha256(data).hexdigest()
    assert schema['bytes'] == len(data)
    assert schema['preview_bytes'] == 65_536
    assert schema['sample_only'] is True
    assert len(json.dumps(result)) < 40_000


@pytest.mark.asyncio
async def test_mcp_inspects_utf16(lab_registry):
    path = lab_registry.get('a').root / 'data' / 'utf16.csv'
    path.write_text('label,value\nstation,3\n', encoding='utf-16')
    async with MCPGateway(lab_registry.get('a')) as gateway:
        result = await gateway.call_tool('inspect_dataset', {'path': 'utf16.csv'})
    assert not result.get('isError'), result
    assert result['structuredContent']['headers'] == ['label', 'value']
    assert result['structuredContent']['encoding'] == 'utf-16'
