# Qianfan API Configuration Guide

This guide explains how to use Qianfan (千帆) platform APIs with the Experience framework.

## Supported LLM Methods

| llm_method | Backend | Qianfan Endpoint |
|------------|---------|------------------|
| `raw_llm_api` | OpenAI SDK | `https://qianfan.baidubce.com/v2` |
| `coding_agent` | Claude Agent SDK | `https://qianfan.baidubce.com/anthropic` |
| `tmux_cc` | ducc CLI | `https://qianfan.baidubce.com/anthropic` |

## Environment Configuration

### Option 1: raw_llm_api (OpenAI Compatible)

```bash
export LLM_BASE_URL="https://qianfan.baidubce.com/v2"
export LLM_API_KEY="your-qianfan-api-key"
export LLM_MODEL="deepseek-v3.2"
```

Usage:
```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method raw_llm_api \
    --num-iterations 1 \
    --total-batch-size 1 \
    --experiment-dir ./my_workspace
```

### Option 2: coding_agent / tmux_cc (Anthropic Compatible)

```bash
export ANTHROPIC_BASE_URL="https://qianfan.baidubce.com/anthropic"
export ANTHROPIC_API_KEY="your-qianfan-api-key"
export ANTHROPIC_MODEL="deepseek-v3.2"
```

Usage:
```bash
# Using coding_agent
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method coding_agent \
    --num-iterations 1 \
    --total-batch-size 1 \
    --experiment-dir ./my_workspace

# Using tmux_cc
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --num-iterations 1 \
    --total-batch-size 1 \
    --experiment-dir ./my_workspace
```

## Available Models on Qianfan

| Model | Recommended For |
|-------|-----------------|
| `deepseek-v3.2` | General purpose, recommended |
| `deepseek-v3.1-250821` | Alternative version |
| `kimi-k2.5` | Fast responses |

## Python API Usage

### raw_llm_api

```python
import os
os.environ['LLM_BASE_URL'] = 'https://qianfan.baidubce.com/v2'
os.environ['LLM_API_KEY'] = 'your-api-key'
os.environ['LLM_MODEL'] = 'deepseek-v3.2'

from experience.llm_client.raw_llm_query import raw_llm_query
import asyncio

result = asyncio.run(raw_llm_query("Hello, world!"))
print(result)
```

### coding_agent

```python
import os
os.environ['ANTHROPIC_BASE_URL'] = 'https://qianfan.baidubce.com/anthropic'
os.environ['ANTHROPIC_API_KEY'] = 'your-api-key'
os.environ['ANTHROPIC_MODEL'] = 'deepseek-v3.2'

from experience.llm_client.coding_agent_query import coding_agent_query
import asyncio

async def main():
    async for message in coding_agent_query(
        prompt="Create a hello.py file",
        cwd='/tmp/demo',
    ):
        print(message)

asyncio.run(main())
```

### Using st_moe_forward

```python
from experience.symbolic_tensor.function.st_moe_forward import st_moe_forward

# With raw_llm_api
output, selected = st_moe_forward(
    input_tensor, 
    experience_tensor,
    llm_method="raw_llm_api",
)

# With coding_agent
output, selected = st_moe_forward(
    input_tensor, 
    experience_tensor,
    llm_method="coding_agent",
)
```

## Troubleshooting

### API Key Format
Qianfan API keys typically start with `bce-v3/`:
```
bce-v3/ALTAK-xxxxx/xxxxxxxxxxxxxxxx
```

### Timeout Issues
For large requests, increase timeout:
```bash
export API_TIMEOUT_MS=600000
```

### Model Not Found
Ensure the model name matches exactly what Qianfan supports. Check the [Qianfan Model Documentation](https://qianfan.baidubce.com) for available models.

## Quick Reference

| Environment Variable | raw_llm_api | coding_agent/tmux_cc |
|---------------------|-------------|----------------------|
| Base URL | `LLM_BASE_URL` | `ANTHROPIC_BASE_URL` |
| API Key | `LLM_API_KEY` | `ANTHROPIC_API_KEY` |
| Model | `LLM_MODEL` | `ANTHROPIC_MODEL` |
