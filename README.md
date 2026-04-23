# Miniforge

High-performance Python library for MiniMax M2.7 inference, optimized for GMKtech M7 hardware (AMD Ryzen 7 PRO 6850H, 28GB RAM).

## Features

- **GGUF Quantization**: Q4_K_M recommended for best quality/size tradeoff
- **TurboQuant KV Cache**: 3-bit compression (turbo3) for 5x smaller memory footprint
- **Tool Calling**: Native support for function calling
- **Vision/Multimodal**: Image understanding capabilities
- **Streaming**: Real-time token streaming for responsive UIs
- **Memory Management**: Hard 28GB limits with automatic optimization
- **Async Support**: Full asyncio support throughout

## Quick Start

```bash
# Install with uv
uv pip install -e .

# Or install from source
git clone https://github.com/Zapdev-labs/miniforge.git
cd miniforge
uv pip install -e ".[all]"
```

The default install omits `llama-cpp-python` so editable installs work on Windows without the Visual Studio C++ toolchain. For the `llama_cpp` GGUF backend, run `uv pip install -e ".[llama-cpp]"` (on Windows you need [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the C++ workload, or a matching prebuilt wheel). The `[all]` extra is server plus dev tools only and does not pull in `llama-cpp-python`.

```python
import asyncio
from miniforge import Miniforge

async def main():
    # Load model (auto-downloads GGUF if available)
    model = await Miniforge.from_pretrained(
        "MiniMaxAI/MiniMax-M2.7",
        quantization="Q4_K_M",
    )
    
    # Simple chat
    response = await model.chat(
        "Explain quantum computing",
        system_prompt="You are a helpful assistant.",
    )
    print(response)
    
    # Streaming
    stream = await model.chat("Tell me a story", stream=True)
    async for token in stream:
        print(token, end="", flush=True)

asyncio.run(main())
```

## Hardware Requirements

**GMKtech M7 Specs:**
- CPU: AMD Ryzen 7 PRO 6850H (8 cores)
- RAM: 28GB available (4GB to iGPU VRAM)
- OS: Windows 11 + WSL2

**Expected Performance:**
- Prompt processing: 50-100 tok/s
- Generation: 15-25 tok/s (exceeds 10 TPS target)
- Memory usage: ~4-5GB with Q4_K_M + turbo3

## Configuration

Create `~/.config/miniforge/config.yaml`:

```yaml
max_memory_gb: 24.0
n_ctx: 200000
quantization: Q4_K_M
cache_type_k: turbo3
cache_type_v: turbo3
n_threads: 8
flash_attn: true
```

Or use the provided optimized config:

```python
from miniforge.utils.config import M7Config

config = M7Config.from_yaml("configs/m7-optimized.yaml")
model = await Miniforge.from_pretrained(config=config)
```

## Examples

See `examples/` directory:

- `basic_chat.py` - Simple chat interface
- `streaming_chat.py` - Real-time streaming
- `tool_agent.py` - Tool calling with custom functions
- `vision_chat.py` - Image understanding

## Backends

### llama.cpp (Recommended)
Fastest CPU inference with GGUF support:

```python
model = await Miniforge.from_pretrained(
    "MiniMaxAI/MiniMax-M2.7",
    backend="llama_cpp",
    quantization="Q4_K_M",
)
```

If no prebuilt GGUF is on the Hub, Miniforge can convert SafeTensors weights automatically using a local [llama.cpp](https://github.com/ggml-org/llama.cpp) checkout: install its Python requirements, build `llama-quantize` (for Q4_K_M etc.), then set `MINIFORGE_LLAMA_CPP` to the repo root (or `llama_cpp_path` in `M7Config` / YAML). The converter runs `convert_hf_to_gguf.py` and caches the result under your miniforge GGUF cache. If conversion is not configured or fails, the library falls back to the Transformers backend as before.

### Transformers (Fallback)
Native HF support with bitsandbytes:

```python
model = await Miniforge.from_pretrained(
    "MiniMaxAI/MiniMax-M2.7",
    backend="transformers",
)
```

## Memory Optimization

The library automatically manages your 28GB constraint:

```python
from miniforge.core.memory import MemoryManager

# Auto-select best quantization
mem = MemoryManager()
quant = mem.select_quantization(model_params=2.7)
# Returns: Q4_K_M (or Q3_K_M if memory constrained)

# Calculate safe context window
max_ctx = mem.calculate_max_context(
    model_quantized_gb=3.1,
    kv_cache_type="turbo3",
)
# Returns: up to 200000 (or less if memory-constrained)
```

## Tool Calling

```python
from miniforge.generation.tools import Tool

weather_tool = Tool(
    name="get_weather",
    description="Get weather for location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"],
    },
    handler=get_weather_func,
)

response = await model.chat(
    "What's the weather in Paris?",
    tools=[weather_tool],
)
```

## Vision

```python
response = await model.chat_vision(
    message="Describe this image",
    image="path/to/image.jpg",
)
```

## License

MIT
