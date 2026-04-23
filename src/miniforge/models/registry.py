"""Model registry for downloading and caching models."""

from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass
import hashlib
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model."""

    id: str
    params_billions: float
    default_quantization: str
    description: str
    tags: List[str]
    vision: bool = False
    tools: bool = True
    max_context: int = 196_608
    is_moe: bool = False


# Known models - organized by model family
ALL_MODELS = {
    # ============================================================================
    # MiniMax M2 Series (229B MoE models)
    # ============================================================================
    # MiniMax M2.7: 228B total params, 256 experts, 8 active per token (~10B active).
    # The "2.7" refers to dense-equivalent active params, NOT total model size.
    # On disk: IQ2_XXS ≈ 61 GB, UD-TQ1_0 ≈ 38 GB. Runs via mmap on 28 GB RAM.
    "MiniMaxAI/MiniMax-M2.7": ModelInfo(
        id="MiniMaxAI/MiniMax-M2.7",
        params_billions=228.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax M2.7 — 228B MoE (256 experts, 8 active). Active params ~10B per token.",
        tags=["multimodal", "vision", "tools", "moe"],
        vision=True,
        tools=True,
        max_context=196_608,
        is_moe=True,
    ),
    "unsloth/MiniMax-M2.7-GGUF": ModelInfo(
        id="unsloth/MiniMax-M2.7-GGUF",
        params_billions=228.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax M2.7 — 228B MoE GGUF via Unsloth. IQ2_XXS ≈ 61 GB on disk.",
        tags=["multimodal", "vision", "tools", "moe"],
        vision=True,
        tools=True,
        max_context=196_608,
        is_moe=True,
    ),
    # MiniMax M2.5: SOTA coding model (80.2% SWE-Bench), agentic tool use
    "MiniMaxAI/MiniMax-M2.5": ModelInfo(
        id="MiniMaxAI/MiniMax-M2.5",
        params_billions=229.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax M2.5 — 229B MoE. SOTA coding (80.2% SWE-Bench), agentic tool use, office automation.",
        tags=["coding", "agents", "tools", "moe"],
        vision=False,
        tools=True,
        max_context=196_608,
        is_moe=True,
    ),
    "unsloth/MiniMax-M2.5-GGUF": ModelInfo(
        id="unsloth/MiniMax-M2.5-GGUF",
        params_billions=229.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax M2.5 — 229B MoE GGUF via Unsloth. SOTA coding with tool calling.",
        tags=["coding", "agents", "tools", "moe"],
        vision=False,
        tools=True,
        max_context=196_608,
        is_moe=True,
    ),
    # MiniMax M2.1: Multilingual coding & multi-task generalization
    "MiniMaxAI/MiniMax-M2.1": ModelInfo(
        id="MiniMaxAI/MiniMax-M2.1",
        params_billions=229.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax M2.1 — 229B MoE. Multilingual coding, multi-task generalization, VIBE benchmark.",
        tags=["coding", "multilingual", "tools", "moe"],
        vision=False,
        tools=True,
        max_context=196_608,
        is_moe=True,
    ),
    "unsloth/MiniMax-M2.1-GGUF": ModelInfo(
        id="unsloth/MiniMax-M2.1-GGUF",
        params_billions=229.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax M2.1 — 229B MoE GGUF via Unsloth. Multilingual coding specialist.",
        tags=["coding", "multilingual", "tools", "moe"],
        vision=False,
        tools=True,
        max_context=196_608,
        is_moe=True,
    ),
    # MiniMax M2: Base model with interleaved thinking, efficient agents
    "MiniMaxAI/MiniMax-M2": ModelInfo(
        id="MiniMaxAI/MiniMax-M2",
        params_billions=229.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax M2 — 229B MoE (~10B active). Interleaved thinking, end-to-end tool use.",
        tags=["coding", "agents", "tools", "moe", "reasoning"],
        vision=False,
        tools=True,
        max_context=196_608,
        is_moe=True,
    ),
    "unsloth/MiniMax-M2-GGUF": ModelInfo(
        id="unsloth/MiniMax-M2-GGUF",
        params_billions=229.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax M2 — 229B MoE GGUF via Unsloth. Base agentic model with reasoning.",
        tags=["coding", "agents", "tools", "moe", "reasoning"],
        vision=False,
        tools=True,
        max_context=196_608,
        is_moe=True,
    ),
    # ============================================================================
    # MiniMax-01 Series (456B Hybrid MoE with Lightning Attention)
    # ============================================================================
    # MiniMax-Text-01: 456B params, 45.9B active, 4M token context, hybrid attention
    "MiniMaxAI/MiniMax-Text-01": ModelInfo(
        id="MiniMaxAI/MiniMax-Text-01",
        params_billions=456.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax-Text-01 — 456B Hybrid MoE (45.9B active). 4M context, Lightning + Softmax attention.",
        tags=["long-context", "tools", "moe", "hybrid-attention"],
        vision=False,
        tools=True,
        max_context=1_000_000,  # 1M training, 4M inference
        is_moe=True,
    ),
    "MiniMaxAI/MiniMax-Text-01-hf": ModelInfo(
        id="MiniMaxAI/MiniMax-Text-01-hf",
        params_billions=456.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax-Text-01 HF — 456B Hybrid MoE. HuggingFace compatible. 4M context.",
        tags=["long-context", "tools", "moe", "hybrid-attention"],
        vision=False,
        tools=True,
        max_context=1_000_000,
        is_moe=True,
    ),
    "unsloth/MiniMax-Text-01-GGUF": ModelInfo(
        id="unsloth/MiniMax-Text-01-GGUF",
        params_billions=456.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax-Text-01 GGUF — 456B Hybrid MoE via Unsloth. 4M context support.",
        tags=["long-context", "tools", "moe", "hybrid-attention"],
        vision=False,
        tools=True,
        max_context=1_000_000,
        is_moe=True,
    ),
    # MiniMax-VL-01: Vision-Language variant of Text-01
    "MiniMaxAI/MiniMax-VL-01": ModelInfo(
        id="MiniMaxAI/MiniMax-VL-01",
        params_billions=456.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax-VL-01 — 456B Hybrid MoE Vision-Language. Multimodal with 4M context.",
        tags=["long-context", "vision", "multimodal", "moe", "hybrid-attention"],
        vision=True,
        tools=True,
        max_context=1_000_000,
        is_moe=True,
    ),
    "unsloth/MiniMax-VL-01-GGUF": ModelInfo(
        id="unsloth/MiniMax-VL-01-GGUF",
        params_billions=456.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax-VL-01 GGUF — 456B Hybrid MoE VLM via Unsloth. Vision + long context.",
        tags=["long-context", "vision", "multimodal", "moe", "hybrid-attention"],
        vision=True,
        tools=True,
        max_context=1_000_000,
        is_moe=True,
    ),
    # ============================================================================
    # MiniMax-M1 Series (456B Reasoning Models with Extended Thinking)
    # ============================================================================
    # MiniMax-M1-40k: Reasoning model with 40K thinking budget
    "MiniMaxAI/MiniMax-M1-40k": ModelInfo(
        id="MiniMaxAI/MiniMax-M1-40k",
        params_billions=456.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax-M1-40k — 456B reasoning model. 40K thinking budget, 55.6% SWE-Bench.",
        tags=["reasoning", "long-context", "tools", "moe", "hybrid-attention"],
        vision=False,
        tools=True,
        max_context=1_000_000,
        is_moe=True,
    ),
    "MiniMaxAI/MiniMax-M1-40k-hf": ModelInfo(
        id="MiniMaxAI/MiniMax-M1-40k-hf",
        params_billions=456.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax-M1-40k HF — 456B reasoning model. HuggingFace compatible.",
        tags=["reasoning", "long-context", "tools", "moe", "hybrid-attention"],
        vision=False,
        tools=True,
        max_context=1_000_000,
        is_moe=True,
    ),
    "unsloth/MiniMax-M1-40k-GGUF": ModelInfo(
        id="unsloth/MiniMax-M1-40k-GGUF",
        params_billions=456.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax-M1-40k GGUF — 456B reasoning model via Unsloth. 40K thinking.",
        tags=["reasoning", "long-context", "tools", "moe", "hybrid-attention"],
        vision=False,
        tools=True,
        max_context=1_000_000,
        is_moe=True,
    ),
    # MiniMax-M1-80k: Reasoning model with 80K thinking budget
    "MiniMaxAI/MiniMax-M1-80k": ModelInfo(
        id="MiniMaxAI/MiniMax-M1-80k",
        params_billions=456.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax-M1-80k — 456B reasoning model. 80K thinking budget, 56.0% SWE-Bench.",
        tags=["reasoning", "long-context", "tools", "moe", "hybrid-attention"],
        vision=False,
        tools=True,
        max_context=1_000_000,
        is_moe=True,
    ),
    "MiniMaxAI/MiniMax-M1-80k-hf": ModelInfo(
        id="MiniMaxAI/MiniMax-M1-80k-hf",
        params_billions=456.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax-M1-80k HF — 456B reasoning model. HuggingFace compatible.",
        tags=["reasoning", "long-context", "tools", "moe", "hybrid-attention"],
        vision=False,
        tools=True,
        max_context=1_000_000,
        is_moe=True,
    ),
    "unsloth/MiniMax-M1-80k-GGUF": ModelInfo(
        id="unsloth/MiniMax-M1-80k-GGUF",
        params_billions=456.0,
        default_quantization="UD-IQ2_XXS",
        description="MiniMax-M1-80k GGUF — 456B reasoning model via Unsloth. 80K thinking.",
        tags=["reasoning", "long-context", "tools", "moe", "hybrid-attention"],
        vision=False,
        tools=True,
        max_context=1_000_000,
        is_moe=True,
    ),
    # ============================================================================
    # Meta Llama 4 Series (2025)
    # ============================================================================
    # Llama 4 Scout: 17B active params, 16 experts, 109B total, multimodal
    "meta-llama/Llama-4-Scout-17B-16E": ModelInfo(
        id="meta-llama/Llama-4-Scout-17B-16E",
        params_billions=109.0,
        default_quantization="Q4_K_M",
        description="Llama 4 Scout — 109B MoE (17B active, 16E). Multimodal, 256K context.",
        tags=["multimodal", "vision", "tools", "moe"],
        vision=True,
        tools=True,
        max_context=262_144,
        is_moe=True,
    ),
    "unsloth/Llama-4-Scout-17B-16E-GGUF": ModelInfo(
        id="unsloth/Llama-4-Scout-17B-16E-GGUF",
        params_billions=109.0,
        default_quantization="Q4_K_M",
        description="Llama 4 Scout GGUF — 109B MoE via Unsloth. Multimodal support.",
        tags=["multimodal", "vision", "tools", "moe"],
        vision=True,
        tools=True,
        max_context=262_144,
        is_moe=True,
    ),
    # Llama 4 Maverick: 17B active params, 128 experts, 400B total, multimodal
    "meta-llama/Llama-4-Maverick-17B-128E": ModelInfo(
        id="meta-llama/Llama-4-Maverick-17B-128E",
        params_billions=400.0,
        default_quantization="Q4_K_M",
        description="Llama 4 Maverick — 400B MoE (17B active, 128E). Advanced multimodal, 256K context.",
        tags=["multimodal", "vision", "tools", "moe"],
        vision=True,
        tools=True,
        max_context=262_144,
        is_moe=True,
    ),
    "unsloth/Llama-4-Maverick-17B-128E-GGUF": ModelInfo(
        id="unsloth/Llama-4-Maverick-17B-128E-GGUF",
        params_billions=400.0,
        default_quantization="Q4_K_M",
        description="Llama 4 Maverick GGUF — 400B MoE via Unsloth. Advanced multimodal.",
        tags=["multimodal", "vision", "tools", "moe"],
        vision=True,
        tools=True,
        max_context=262_144,
        is_moe=True,
    ),
    # Llama 3.3 70B: Dense model, improved performance
    "meta-llama/Llama-3.3-70B": ModelInfo(
        id="meta-llama/Llama-3.3-70B",
        params_billions=70.0,
        default_quantization="Q4_K_M",
        description="Llama 3.3 70B — Dense model. Improved multilingual and reasoning.",
        tags=["multilingual", "reasoning", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "unsloth/Llama-3.3-70B-GGUF": ModelInfo(
        id="unsloth/Llama-3.3-70B-GGUF",
        params_billions=70.0,
        default_quantization="Q4_K_M",
        description="Llama 3.3 70B GGUF — Dense model via Unsloth.",
        tags=["multilingual", "reasoning", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    # Llama 3.2 Vision (Multimodal small models)
    "meta-llama/Llama-3.2-11B-Vision": ModelInfo(
        id="meta-llama/Llama-3.2-11B-Vision",
        params_billions=11.0,
        default_quantization="Q4_K_M",
        description="Llama 3.2 11B Vision — Multimodal small model.",
        tags=["vision", "multimodal"],
        vision=True,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "meta-llama/Llama-3.2-90B-Vision": ModelInfo(
        id="meta-llama/Llama-3.2-90B-Vision",
        params_billions=90.0,
        default_quantization="Q4_K_M",
        description="Llama 3.2 90B Vision — Large multimodal model.",
        tags=["vision", "multimodal"],
        vision=True,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "unsloth/Llama-3.2-11B-Vision-GGUF": ModelInfo(
        id="unsloth/Llama-3.2-11B-Vision-GGUF",
        params_billions=11.0,
        default_quantization="Q4_K_M",
        description="Llama 3.2 11B Vision GGUF via Unsloth.",
        tags=["vision", "multimodal"],
        vision=True,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    # ============================================================================
    # Mistral AI Models
    # ============================================================================
    # Mistral Small 3.1: 24B dense, multimodal
    "mistralai/Mistral-Small-3.1-24B": ModelInfo(
        id="mistralai/Mistral-Small-3.1-24B",
        params_billions=24.0,
        default_quantization="Q4_K_M",
        description="Mistral Small 3.1 — 24B dense. Multimodal, 128K context.",
        tags=["multimodal", "vision", "tools"],
        vision=True,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "unsloth/Mistral-Small-3.1-24B-GGUF": ModelInfo(
        id="unsloth/Mistral-Small-3.1-24B-GGUF",
        params_billions=24.0,
        default_quantization="Q4_K_M",
        description="Mistral Small 3.1 GGUF — 24B via Unsloth. Multimodal.",
        tags=["multimodal", "vision", "tools"],
        vision=True,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    # Mistral Large 2: 123B total, 32B active MoE
    "mistralai/Mistral-Large-2": ModelInfo(
        id="mistralai/Mistral-Large-2",
        params_billions=123.0,
        default_quantization="Q4_K_M",
        description="Mistral Large 2 — 123B MoE (32B active). 128K context.",
        tags=["multilingual", "tools", "moe"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=True,
    ),
    "unsloth/Mistral-Large-2-GGUF": ModelInfo(
        id="unsloth/Mistral-Large-2-GGUF",
        params_billions=123.0,
        default_quantization="Q4_K_M",
        description="Mistral Large 2 GGUF — 123B MoE via Unsloth.",
        tags=["multilingual", "tools", "moe"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=True,
    ),
    # Mistral Nemo: 12B dense, instruction-tuned
    "mistralai/Mistral-Nemo-12B": ModelInfo(
        id="mistralai/Mistral-Nemo-12B",
        params_billions=12.0,
        default_quantization="Q4_K_M",
        description="Mistral Nemo — 12B dense. Efficient instruction model.",
        tags=["instruction", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "unsloth/Mistral-Nemo-12B-GGUF": ModelInfo(
        id="unsloth/Mistral-Nemo-12B-GGUF",
        params_billions=12.0,
        default_quantization="Q4_K_M",
        description="Mistral Nemo GGUF — 12B via Unsloth.",
        tags=["instruction", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    # ============================================================================
    # Alibaba Qwen Models
    # ============================================================================
    # Qwen3 Series: Various sizes 0.6B to 235B A3B MoE
    "Qwen/Qwen3-235B-A22B": ModelInfo(
        id="Qwen/Qwen3-235B-A22B",
        params_billions=235.0,
        default_quantization="Q4_K_M",
        description="Qwen3-235B-A22B — 235B MoE (22B active). Advanced reasoning.",
        tags=["reasoning", "tools", "moe"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=True,
    ),
    "Qwen/Qwen3-32B": ModelInfo(
        id="Qwen/Qwen3-32B",
        params_billions=32.0,
        default_quantization="Q4_K_M",
        description="Qwen3-32B — Dense model. Strong performance, 128K context.",
        tags=["reasoning", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "Qwen/Qwen3-14B": ModelInfo(
        id="Qwen/Qwen3-14B",
        params_billions=14.0,
        default_quantization="Q4_K_M",
        description="Qwen3-14B — Dense model. Balanced efficiency.",
        tags=["reasoning", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "Qwen/Qwen3-8B": ModelInfo(
        id="Qwen/Qwen3-8B",
        params_billions=8.0,
        default_quantization="Q4_K_M",
        description="Qwen3-8B — Dense model. Efficient local inference.",
        tags=["reasoning", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "Qwen/Qwen3-4B": ModelInfo(
        id="Qwen/Qwen3-4B",
        params_billions=4.0,
        default_quantization="Q4_K_M",
        description="Qwen3-4B — Dense model. Lightweight.",
        tags=["reasoning", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "Qwen/Qwen3-1.7B": ModelInfo(
        id="Qwen/Qwen3-1.7B",
        params_billions=1.7,
        default_quantization="Q4_K_M",
        description="Qwen3-1.7B — Dense model. Edge deployment.",
        tags=["reasoning", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "Qwen/Qwen3-0.6B": ModelInfo(
        id="Qwen/Qwen3-0.6B",
        params_billions=0.6,
        default_quantization="Q4_K_M",
        description="Qwen3-0.6B — Dense model. Ultra-lightweight.",
        tags=["reasoning", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "unsloth/Qwen3-GGUF": ModelInfo(
        id="unsloth/Qwen3-GGUF",
        params_billions=32.0,
        default_quantization="Q4_K_M",
        description="Qwen3 GGUF via Unsloth. Various sizes available.",
        tags=["reasoning", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    # QwQ: Reasoning models
    "Qwen/QwQ-32B": ModelInfo(
        id="Qwen/QwQ-32B",
        params_billions=32.0,
        default_quantization="Q4_K_M",
        description="QwQ-32B — Reasoning specialist. Deep thinking capabilities.",
        tags=["reasoning", "math", "coding", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "unsloth/QwQ-32B-GGUF": ModelInfo(
        id="unsloth/QwQ-32B-GGUF",
        params_billions=32.0,
        default_quantization="Q4_K_M",
        description="QwQ-32B GGUF via Unsloth. Reasoning specialist.",
        tags=["reasoning", "math", "coding", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    # Qwen2.5 VL: Vision-Language
    "Qwen/Qwen2.5-VL-72B": ModelInfo(
        id="Qwen/Qwen2.5-VL-72B",
        params_billions=72.0,
        default_quantization="Q4_K_M",
        description="Qwen2.5-VL-72B — Large vision-language model.",
        tags=["vision", "multimodal", "tools"],
        vision=True,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "Qwen/Qwen2.5-VL-32B": ModelInfo(
        id="Qwen/Qwen2.5-VL-32B",
        params_billions=32.0,
        default_quantization="Q4_K_M",
        description="Qwen2.5-VL-32B — Vision-language model.",
        tags=["vision", "multimodal", "tools"],
        vision=True,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "Qwen/Qwen2.5-VL-7B": ModelInfo(
        id="Qwen/Qwen2.5-VL-7B",
        params_billions=7.0,
        default_quantization="Q4_K_M",
        description="Qwen2.5-VL-7B — Efficient vision-language model.",
        tags=["vision", "multimodal", "tools"],
        vision=True,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "unsloth/Qwen2.5-VL-GGUF": ModelInfo(
        id="unsloth/Qwen2.5-VL-GGUF",
        params_billions=32.0,
        default_quantization="Q4_K_M",
        description="Qwen2.5-VL GGUF via Unsloth. Vision-language.",
        tags=["vision", "multimodal", "tools"],
        vision=True,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    # Qwen2.5 Coder
    "Qwen/Qwen2.5-Coder-32B": ModelInfo(
        id="Qwen/Qwen2.5-Coder-32B",
        params_billions=32.0,
        default_quantization="Q4_K_M",
        description="Qwen2.5-Coder-32B — Code generation specialist.",
        tags=["coding", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "Qwen/Qwen2.5-Coder-14B": ModelInfo(
        id="Qwen/Qwen2.5-Coder-14B",
        params_billions=14.0,
        default_quantization="Q4_K_M",
        description="Qwen2.5-Coder-14B — Code generation specialist.",
        tags=["coding", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "Qwen/Qwen2.5-Coder-7B": ModelInfo(
        id="Qwen/Qwen2.5-Coder-7B",
        params_billions=7.0,
        default_quantization="Q4_K_M",
        description="Qwen2.5-Coder-7B — Efficient code generation.",
        tags=["coding", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    "unsloth/Qwen2.5-Coder-GGUF": ModelInfo(
        id="unsloth/Qwen2.5-Coder-GGUF",
        params_billions=32.0,
        default_quantization="Q4_K_M",
        description="Qwen2.5-Coder GGUF via Unsloth. Code specialist.",
        tags=["coding", "tools"],
        vision=False,
        tools=True,
        max_context=131_072,
        is_moe=False,
    ),
    # Kimi K2.5: 1T-param MoE, 384 experts, 8 active per token (2% sparsity).
    # mmap makes this viable on 28 GB RAM — only active experts page from disk.
    "bakosh/Huihui-Kimi-K2.5-BF16-abliterated-GGUF": ModelInfo(
        id="bakosh/Huihui-Kimi-K2.5-BF16-abliterated-GGUF",
        params_billions=1000.0,
        default_quantization="Q2_K",
        description="Kimi K2.5 1T MoE abliterated GGUF — 384 experts, 8 active per token. Q2_K=373 GB, Q3_K=550 GB.",
        tags=["reasoning", "chat", "vision", "tools", "moe"],
        vision=True,
        tools=True,
        max_context=262_144,
        is_moe=True,
    ),
    "unsloth/Kimi-K2.5-GGUF": ModelInfo(
        id="unsloth/Kimi-K2.5-GGUF",
        params_billions=1000.0,
        default_quantization="UD-TQ1_0",
        description="Kimi K2.5 1T MoE via Unsloth — smallest quant UD-TQ1_0 = 240 GB on disk.",
        tags=["reasoning", "chat", "vision", "tools", "moe"],
        vision=True,
        tools=True,
        max_context=262_144,
        is_moe=True,
    ),
    # ============================================================================
    # Transferred Local Models (192.168.1.46 - miniforge server)
    # ============================================================================
    # MiniMax-M2.7-161B-REAP-Q3_K_M: 73GB single-file GGUF (transferred to remote)
    "local/MiniMax-M2.7-161B-REAP-Q3_K_M": ModelInfo(
        id="local/MiniMax-M2.7-161B-REAP-Q3_K_M",
        params_billions=161.0,
        default_quantization="Q3_K_M",
        description="MiniMax-M2.7-161B-REAP at Q3_K_M — 73GB. Pruned 161B MoE, higher quality than Q2_K.",
        tags=["pruned", "moe", "tools", "reasoning"],
        vision=False,
        tools=True,
        max_context=196_608,
        is_moe=True,
    ),
    # Kimi-K2.5-UD-TQ1_0: 224GB 5-shard GGUF (transferred to remote)
    "local/Kimi-K2.5-UD-TQ1_0": ModelInfo(
        id="local/Kimi-K2.5-UD-TQ1_0",
        params_billions=1000.0,
        default_quantization="UD-TQ1_0",
        description="Kimi-K2.5 1T MoE at UD-TQ1_0 — 224GB, 5 shards. Ultra-high quality quantization.",
        tags=["reasoning", "chat", "vision", "tools", "moe"],
        vision=True,
        tools=True,
        max_context=262_144,
        is_moe=True,
    ),
}


class ModelRegistry:
    """
    Manage model downloads and conversions.

    Handles:
    - HuggingFace model downloads
    - GGUF conversion caching
    - Model metadata
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "miniforge"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.gguf_dir = self.cache_dir / "gguf"
        self.gguf_dir.mkdir(exist_ok=True)
        self.hf_dir = self.cache_dir / "huggingface"
        self.hf_dir.mkdir(exist_ok=True)

    @staticmethod
    def hf_hub_models_root() -> Path:
        """Directory containing ``models--org--name`` (same tree as Transformers / ``hf`` CLI)."""
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            return Path(HF_HUB_CACHE)
        except Exception:
            hf_home = os.environ.get("HF_HOME")
            base = Path(hf_home).expanduser() if hf_home else Path.home() / ".cache" / "huggingface"
            return base / "hub"

    def model_id_to_cache_stem(self, model_id: str) -> str:
        """Stable filename stem for GGUF cache entries (HF repo id or hashed local dir)."""
        p = Path(model_id).expanduser()
        try:
            if p.is_dir():
                digest = hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:32]
                return f"local-{digest}"
        except OSError:
            pass
        return model_id.replace("\\", "--").replace("/", "--")

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a known model, with fuzzy matching for GGUF repo variants."""
        info = ALL_MODELS.get(model_id)
        if info is not None:
            return info

        # Fuzzy match: check if model_id contains a known model's base name
        # e.g. "unsloth/MiniMax-M2.7-GGUF" should match "MiniMax-M2.7"
        model_id_lower = model_id.lower()
        for key, model_info in ALL_MODELS.items():
            # Extract base model name from the key (after org/)
            base_name = key.split("/")[-1].lower()
            if base_name in model_id_lower:
                logger.info("Fuzzy-matched model %s -> %s", model_id, key)
                return model_info

        return None

    def get_cached_gguf_path(
        self,
        model_id: str,
        quantization: str,
    ) -> Optional[Path]:
        """
        Get path to cached GGUF model if it exists.

        Args:
            model_id: HuggingFace model ID
            quantization: Quantization type (Q4_K_M, etc.)

        Returns:
            Path if cached, None otherwise
        """
        safe_name = self.model_id_to_cache_stem(model_id)
        filename = f"{safe_name}_{quantization}.gguf"
        cache_path = self.gguf_dir / filename

        if cache_path.exists():
            return cache_path

        return None

    def register_gguf(
        self,
        model_id: str,
        quantization: str,
        gguf_path: Path,
    ) -> Path:
        """
        Register an external GGUF file in the cache.

        Args:
            model_id: Original model ID
            quantization: Quantization type
            gguf_path: Path to GGUF file

        Returns:
            Path in cache
        """
        safe_name = self.model_id_to_cache_stem(model_id)
        filename = f"{safe_name}_{quantization}.gguf"
        cache_path = self.gguf_dir / filename

        # Copy or symlink
        if gguf_path != cache_path:
            import shutil

            shutil.copy2(gguf_path, cache_path)
            logger.info(f"Cached GGUF: {cache_path}")

        return cache_path

    def download_hf_model(
        self,
        model_id: str,
        local_files_only: bool = False,
    ) -> Path:
        """
        Resolve or download weights using the shared Hugging Face hub cache
        (e.g. ``%USERPROFILE%\\.cache\\huggingface\\hub\\models--Org--Name`` on Windows).

        Returns the snapshot directory (config + weights) suitable for conversion or inspection.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface-hub not installed. Install with: uv pip install huggingface-hub"
            )

        logger.info("Resolving %s in Hugging Face hub cache...", model_id)
        out = snapshot_download(
            repo_id=model_id,
            local_files_only=local_files_only,
        )
        return Path(out)

    def find_gguf_in_repo(self, repo_path: Path) -> Optional[Path]:
        """
        Find GGUF files in a repository.

        Args:
            repo_path: Path to model repository

        Returns:
            Path to GGUF file if found, None otherwise
        """
        # Search for GGUF files
        gguf_files = list(repo_path.rglob("*.gguf"))

        if not gguf_files:
            return None

        # Prefer specific quantization
        preferred_order = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "Q3_K_M", "Q4_0"]

        for quant in preferred_order:
            for f in gguf_files:
                if quant in f.name:
                    return f

        # Return first GGUF
        return gguf_files[0]

    def list_models(self, family: Optional[str] = None) -> List[ModelInfo]:
        """List all known models, optionally filtered by family.

        Args:
            family: Optional filter - "minimax", "llama", "mistral", "qwen"

        Returns:
            List of ModelInfo objects
        """
        models = list(ALL_MODELS.values())

        if family:
            family_lower = family.lower()
            family_map = {
                "minimax": ["minimax"],
                "llama": ["llama"],
                "mistral": ["mistral"],
                "qwen": ["qwen", "qwq"],
                "kimi": ["kimi"],
            }
            keywords = family_map.get(family_lower, [family_lower])
            models = [
                m for m in models
                if any(kw in m.id.lower() for kw in keywords)
            ]

        return models

    def list_cached_models(self) -> List[str]:
        """List all cached models."""
        models = []

        # List GGUF models
        for gguf_file in self.gguf_dir.glob("*.gguf"):
            name = gguf_file.stem.replace("--", "/")
            models.append(f"[GGUF] {name}")

        hub = self.hf_hub_models_root()
        if hub.is_dir():
            for entry in sorted(hub.glob("models--*")):
                if entry.is_dir():
                    rest = entry.name.removeprefix("models--")
                    repo_id = rest.replace("--", "/", 1) if "--" in rest else rest
                    models.append(f"[HF hub] {repo_id}")

        for legacy in self.hf_dir.iterdir():
            if legacy.is_dir():
                name = legacy.name.replace("--", "/")
                models.append(f"[HF legacy] {name}")

        return models

    def clear_cache(self, confirm: bool = False) -> None:
        """Clear model cache."""
        if not confirm:
            print("Set confirm=True to clear cache")
            return

        import shutil

        if self.gguf_dir.exists():
            shutil.rmtree(self.gguf_dir)
            self.gguf_dir.mkdir(exist_ok=True)

        if self.hf_dir.exists():
            shutil.rmtree(self.hf_dir)
            self.hf_dir.mkdir(exist_ok=True)

        logger.info("Model cache cleared")


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_registry(cache_dir: Optional[Path] = None) -> ModelRegistry:
    """Get global model registry instance."""
    global _global_registry

    if _global_registry is None:
        _global_registry = ModelRegistry(cache_dir)

    return _global_registry
