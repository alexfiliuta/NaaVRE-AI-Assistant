"""Measure logged phases for ONE notebook build; never infer execution time."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re


def summarize(text):
    events = []
    for line in text.splitlines():
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+ServerApp\]\s+(.*)', line)
        if match:
            events.append((datetime.fromisoformat(match[1]), match[2]))
    plans = [t for t, m in events if 'planning attempt 1/' in m and 'started' in m]
    if len(plans) != 1:
        raise ValueError('Provide a log containing exactly one build (one planning attempt 1).')
    start = plans[0]
    cells = [t for t, m in events if t >= start and 'generating cell ' in m]
    responses = [(t, m) for t, m in events if t >= start and
                 'local model response received: kind=cell ' in m]
    if not cells or not responses:
        raise ValueError('Build has no recorded cell generation/response completion.')
    end = responses[-1][0]
    def seconds(a, b):
        return round((b-a).total_seconds(), 3)
    result = {'planning_start': start.isoformat(), 'last_cell_response': end.isoformat(),
              'build_to_last_model_response_seconds': seconds(start, end),
              'planning_until_first_cell_seconds': seconds(start, cells[0]),
              'cell_generation_span_seconds': seconds(cells[0], end),
              'cell_response_count_including_retries': len(responses),
              'planning_rejections': sum('plan rejected' in m for t, m in events if start <= t <= end),
              'execution_time_seconds': None,
              'limitation': 'Server-log phases only; not exact UI insertion completion or notebook execution.'}
    agent_start = [t for t, m in events if t < start and
                   'request started: kind=agent decision' in m]
    agent_end = [t for t, m in events if t < start and
                 'response received: kind=agent decision' in m]
    if agent_start and agent_end:
        result.update({'observed_investigation_model_span_seconds': seconds(agent_start[0], agent_end[-1]),
                       'investigation_to_build_gap_seconds': seconds(agent_end[-1], start),
                       'combined_wall_span_including_gap_seconds': seconds(agent_start[0], end)})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.log.read_text(encoding='utf-8-sig'))
    result['source_log'] = str(args.log.resolve())
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as f:
            f.write(encoded)
    print(encoded)
