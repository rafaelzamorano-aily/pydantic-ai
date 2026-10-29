"""Usage accounting across Anthropic-compatible gateway response shapes.

Synthetic HTTP responses reproduce LiteLLM's Bedrock usage promotion, which cannot be
requested on demand from Anthropic. The real SDK parses every response and stream.
"""

from __future__ import annotations as _annotations

import json
from typing import Literal

import httpx2
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent

from ...conftest import try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


@pytest.mark.parametrize('mode', ['nonstream', 'implicit', 'explicit'])
@pytest.mark.parametrize('aggregate_top_level', [False, True])
@pytest.mark.parametrize('producer_1h', [0, 1500])
async def test_compaction_usage_uses_iteration_totals(
    allow_model_requests: None,
    mode: Literal['nonstream', 'implicit', 'explicit'],
    aggregate_top_level: bool,
    producer_1h: int,
) -> None:
    """Compaction is billed once, with each pass's actual cache TTL, on all request paths.

    The counts reproduce a captured LiteLLM/Bedrock response, plus a producer one-hour
    cache variant. The start event carries the compactor's cache split; the final delta
    reports the producer's split only inside `iterations`.
    """
    compaction = {
        'type': 'compaction',
        'input_tokens': 46,
        'output_tokens': 1212,
        'cache_creation_input_tokens': 55503,
        'cache_read_input_tokens': 1226,
        'cache_creation': {'ephemeral_1h_input_tokens': 55503, 'ephemeral_5m_input_tokens': 0},
    }
    producer = {
        'type': 'message',
        'input_tokens': 0,
        'output_tokens': 40,
        'cache_creation_input_tokens': 2211,
        'cache_read_input_tokens': 0,
        'cache_creation': {'ephemeral_1h_input_tokens': producer_1h, 'ephemeral_5m_input_tokens': 2211 - producer_1h},
    }
    final_usage = {
        'input_tokens': 2257 if aggregate_top_level else 0,
        'output_tokens': 40,
        'cache_creation_input_tokens': 55503 if aggregate_top_level else 2211,
        'cache_read_input_tokens': 1226 if aggregate_top_level else 0,
        'iterations': [compaction, producer],
    }
    message = {
        'id': 'msg_usage',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-5',
        'content': [{'type': 'text', 'text': 'Hello'}],
        'stop_reason': 'end_turn',
        'stop_sequence': None,
        'usage': final_usage,
    }
    events = [
        {
            'type': 'message_start',
            'message': {
                **message,
                'content': [],
                'stop_reason': None,
                'usage': {key: value for key, value in compaction.items() if key != 'type'},
            },
        },
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hello'}},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': final_usage},
        {'type': 'message_stop'},
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        assert body.get('stream', False) is (mode != 'nonstream')
        if mode == 'nonstream':
            return httpx2.Response(200, json=message)
        content = ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events)
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=content)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        client = AsyncAnthropic(api_key='test', http_client=http_client)
        agent = Agent(
            AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client)),
            model_settings=AnthropicModelSettings(max_tokens=32768 if mode == 'implicit' else 1024),
        )
        if mode == 'explicit':
            async with agent.run_stream('Hello') as result:
                assert await result.get_output() == 'Hello'
                usage = result.usage
        else:
            result = await agent.run('Hello')
            assert result.output == 'Hello'
            usage = result.usage

    assert (usage.input_tokens, usage.output_tokens, usage.cache_write_tokens, usage.cache_read_tokens) == snapshot(
        (58986, 1252, 57714, 1226)
    )
    assert usage.cache_write_1h_tokens == 55503 + producer_1h
    assert usage.details['compaction_iterations'] == 1
    assert usage.details['message_iterations'] == 1


@pytest.mark.parametrize(
    ('pass_types', 'expected_tokens', 'expected_messages'),
    [
        (['compaction'], (10, 2), 0),
        (['compaction', 'message', 'compaction', 'fallback_message'], (40, 8), 2),
        (['compaction', 'advisor_message', 'message'], (20, 4), 1),
    ],
)
async def test_compaction_usage_excludes_advisor_and_counts_all_executor_passes(
    allow_model_requests: None,
    pass_types: list[str],
    expected_tokens: tuple[int, int],
    expected_messages: int,
) -> None:
    """Synthetic uncached iterations pin multiple passes, fallback messages and compaction-only usage.

    Advisor tokens keep their separate pricing details instead of entering executor billing.
    """
    message = {
        'id': 'msg_usage',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-5',
        'content': [{'type': 'text', 'text': 'Hello'}],
        'stop_reason': 'end_turn',
        'stop_sequence': None,
        'usage': {
            'input_tokens': 999999,
            'output_tokens': 999999,
            'iterations': [
                {
                    'type': pass_type,
                    'input_tokens': 10,
                    'output_tokens': 2,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                }
                for pass_type in pass_types
            ],
        },
    }
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=message))
    ) as http_client:
        client = AsyncAnthropic(api_key='test', http_client=http_client)
        agent = Agent(
            AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client)),
            model_settings=AnthropicModelSettings(max_tokens=1024),
        )
        result = await agent.run('Hello')
    assert (result.usage.input_tokens, result.usage.output_tokens) == expected_tokens
    assert result.usage.cache_read_tokens == result.usage.cache_write_tokens == result.usage.cache_write_1h_tokens == 0
    assert result.usage.details['message_iterations'] == expected_messages
    if 'advisor_message' in pass_types:
        assert result.usage.details['advisor_input_tokens'] == 10
        assert result.usage.details['advisor_output_tokens'] == 2
