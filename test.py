"""Quick local Gemma GGUF smoke test using llama-cpp-python."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from llama_cpp import Llama

DEFAULT_MODEL_DIR = Path(r"D:\AI\models\freakyskittle\gemma-4-9b-it-Q2_K")


def find_gguf(model_dir: Path) -> Path:
    """Return the first GGUF file in a model directory."""
    if model_dir.is_file() and model_dir.suffix.lower() == ".gguf":
        return model_dir

    gguf_files = sorted(model_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf files found in {model_dir}")
    return gguf_files[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a quick Gemma GGUF generation test.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", default="Write a short hello from Gemma in one paragraph.")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = find_gguf(args.model)

    print(f"Loading Gemma GGUF: {model_path}")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.ctx,
        n_threads=args.threads,
        n_batch=args.batch,
        n_gpu_layers=args.gpu_layers,
        flash_attn=True,
        use_mmap=True,
        verbose=args.verbose,
    )

    print("\nPrompt:")
    print(args.prompt)
    print("\nResponse:")

    stream = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": args.prompt},
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=True,
    )

    start_time = time.perf_counter()
    generated_tokens = 0
    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        text = delta.get("content") or ""
        if text:
            generated_tokens += 1
            print(text, end="", flush=True)

    elapsed = time.perf_counter() - start_time
    tps = generated_tokens / elapsed if elapsed > 0 else 0.0
    print(f"\n\nTPS: {tps:.2f} tokens/sec ({generated_tokens} tokens in {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
