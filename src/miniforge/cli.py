"""Command-line interface for Miniforge."""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from miniforge.models.registry import get_registry
from miniforge.utils.config import M7Config


def _build_config(args: argparse.Namespace) -> M7Config:
    """Build runtime config from preset, environment, and CLI overrides."""
    base = M7Config.performance_preset(args.preset) if getattr(args, "preset", None) else None
    config = M7Config.from_env(base=base)
    config.apply_overrides(
        model_id=getattr(args, "model", None),
        backend=getattr(args, "backend", None),
        quantization=getattr(args, "quantization", None),
        download_dir=getattr(args, "download_dir", None),
        max_tokens=getattr(args, "max_tokens", None),
        temperature=getattr(args, "temperature", None),
        top_p=getattr(args, "top_p", None),
    )
    return config


def _add_runtime_arguments(parser: argparse.ArgumentParser, *, include_generation: bool) -> None:
    """Add shared runtime arguments to a subcommand parser."""
    parser.add_argument("--model", "-m", help="Model ID or path")
    parser.add_argument("--backend", help="Backend to use")
    parser.add_argument("--quantization", "-q", help="Quantization type")
    parser.add_argument(
        "--preset",
        choices=["speed", "balanced", "memory", "quality", "moe"],
        help="Performance preset",
    )
    parser.add_argument(
        "--download-dir",
        "-d",
        help="Directory to store/load GGUF files (e.g. D:/AI)",
    )
    if include_generation:
        parser.add_argument(
            "--system-prompt",
            default="You are a helpful assistant.",
            help="System prompt for chat sessions",
        )
        parser.add_argument("--max-tokens", type=int, help="Override generation max tokens")
        parser.add_argument("--temperature", type=float, help="Override sampling temperature")
        parser.add_argument("--top-p", type=float, help="Override nucleus sampling")


def cmd_chat(args):
    """Run interactive chat."""
    from miniforge import Miniforge

    async def interactive():
        config = _build_config(args)
        model_id = config.model_id
        print(f"Loading {model_id}...")
        model = await Miniforge.from_pretrained(
            model_id,
            quantization=config.quantization,
            config=config,
            backend=config.backend,
            download_dir=config.download_dir,
        )
        print(
            "Loaded "
            f"{model_id} [{config.backend}, {config.quantization}, {config.n_ctx} ctx]. "
            "Type 'quit' to exit.\n"
        )

        history = []

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                t0 = time.perf_counter()
                response = await model.chat(
                    message=user_input,
                    history=history,
                    system_prompt=args.system_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stream=args.stream,
                )

                if args.stream:
                    print("Assistant: ", end="", flush=True)
                    n_stream_tokens = 0
                    async for token in response:
                        print(token, end="", flush=True)
                        n_stream_tokens += 1
                    elapsed = time.perf_counter() - t0
                    tps = n_stream_tokens / elapsed if elapsed > 0 else 0
                    print(f"\n[{n_stream_tokens} tokens, {elapsed:.1f}s, {tps:.2f} tok/s]\n")
                else:
                    elapsed = time.perf_counter() - t0
                    n_words = len(response.split()) if isinstance(response, str) else 0
                    tps = n_words / elapsed if elapsed > 0 else 0
                    print(f"Assistant: {response}")
                    print(f"[~{n_words} tokens, {elapsed:.1f}s, ~{tps:.2f} tok/s]\n")

                history.append({"role": "user", "content": user_input})
                history.append(
                    {
                        "role": "assistant",
                        "content": response if isinstance(response, str) else "[streamed]",
                    }
                )

            except KeyboardInterrupt:
                break

        await model.cleanup()
        print("\nGoodbye!")

    asyncio.run(interactive())


def cmd_server(args):
    """Run API server."""
    from miniforge.webui.server import run_server

    run_server(
        host=args.host,
        port=args.port,
        model=args.model,
        backend=args.backend,
        quantization=args.quantization,
        download_dir=args.download_dir,
        preset=args.preset,
    )


def cmd_webui(args):
    """Launch full WebUI stack (API + OpenWebUI + Grafana)."""
    import subprocess

    env = os.environ.copy()
    if args.model:
        env["MINIFORGE_MODEL"] = args.model
    if args.backend:
        env["MINIFORGE_BACKEND"] = args.backend
    if args.quantization:
        env["MINIFORGE_QUANTIZATION"] = args.quantization
    if args.preset:
        env["MINIFORGE_PRESET"] = args.preset
    env["MINIFORGE_PORT"] = str(args.port)
    env["MINIFORGE_HOST"] = args.host
    if args.download_dir:
        env["MINIFORGE_DOWNLOAD_DIR"] = args.download_dir

    start_script = Path(__file__).parent.parent.parent / "start.sh"
    if not start_script.exists():
        print(f"start.sh not found at {start_script}")
        sys.exit(1)

    subprocess.run([str(start_script)], env=env)


def cmd_download(args):
    """Download model."""
    registry = get_registry()

    print(f"Downloading {args.model}...")

    try:
        path = registry.download_hf_model(args.model)
        print(f"Downloaded to: {path}")

        gguf = registry.find_gguf_in_repo(path)
        if gguf:
            print(f"Found GGUF: {gguf}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list(args):
    """List cached models."""
    registry = get_registry()
    models = registry.list_cached_models()

    if not models:
        print("No cached models found.")
    else:
        print("Cached models:")
        for model in models:
            print(f"  - {model}")


def cmd_doctor(args):
    """Show detected hardware and resolved runtime config."""
    from miniforge.utils.hardware import detect_hardware

    profile = detect_hardware()
    base = M7Config.performance_preset(args.preset) if args.preset else None
    config = M7Config.from_env(base=base)

    payload = {
        "hardware": {
            "os": profile.os_name,
            "is_wsl": profile.is_wsl,
            "cpu": {
                "brand": profile.cpu.brand,
                "physical_cores": profile.cpu.physical_cores,
                "logical_cores": profile.cpu.logical_cores,
                "flags": profile.cpu.flags,
            },
            "ram_gb": {
                "total": round(profile.total_ram_gb, 2),
                "available": round(profile.available_ram_gb, 2),
            },
            "gpus": [
                {
                    "name": gpu.name,
                    "vendor": gpu.vendor,
                    "vram_gb": gpu.vram_gb,
                }
                for gpu in profile.gpus
            ],
        },
        "config": config.summary(),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("Hardware")
    print(f"  OS: {payload['hardware']['os']} (WSL={payload['hardware']['is_wsl']})")
    print(
        "  CPU: "
        f"{payload['hardware']['cpu']['brand']} "
        f"[{payload['hardware']['cpu']['physical_cores']}C/"
        f"{payload['hardware']['cpu']['logical_cores']}T]"
    )
    print(
        "  RAM: "
        f"{payload['hardware']['ram_gb']['total']} GB total, "
        f"{payload['hardware']['ram_gb']['available']} GB available"
    )
    if payload["hardware"]["gpus"]:
        print("  GPUs:")
        for gpu in payload["hardware"]["gpus"]:
            print(f"    - {gpu['vendor']} {gpu['name']} ({gpu['vram_gb']} GB)")
    else:
        print("  GPUs: none detected")

    print("\nResolved runtime")
    summary = payload["config"]
    print(
        f"  Model: {summary['model_id']}\n"
        f"  Backend: {summary['backend']}\n"
        f"  Quantization: {summary['quantization']}\n"
        f"  Context: {summary['n_ctx']}\n"
        f"  Threads: {summary['n_threads']}\n"
        f"  Generation: max_tokens={summary['generation']['max_tokens']}, "
        f"temperature={summary['generation']['temperature']}, top_p={summary['generation']['top_p']}"
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="miniforge",
        description="Miniforge: High-performance local inference",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    _add_runtime_arguments(chat_parser, include_generation=True)
    chat_parser.add_argument("--stream", "-s", action="store_true", help="Stream output")
    chat_parser.set_defaults(func=cmd_chat)

    # Server command
    server_parser = subparsers.add_parser("serve", help="Run API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    server_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind")
    _add_runtime_arguments(server_parser, include_generation=False)
    server_parser.set_defaults(func=cmd_server)

    # WebUI command (full stack)
    webui_parser = subparsers.add_parser("webui", help="Launch API + OpenWebUI + Grafana")
    webui_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    webui_parser.add_argument("--port", "-p", type=int, default=8000, help="API port to bind")
    _add_runtime_arguments(webui_parser, include_generation=False)
    webui_parser.set_defaults(func=cmd_webui)

    # Download command
    download_parser = subparsers.add_parser("download", help="Download model")
    download_parser.add_argument("model", help="Model ID to download")
    download_parser.set_defaults(func=cmd_download)

    # List command
    list_parser = subparsers.add_parser("list", help="List cached models")
    list_parser.set_defaults(func=cmd_list)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Show resolved hardware and runtime config"
    )
    doctor_parser.add_argument(
        "--preset",
        choices=["speed", "balanced", "memory", "quality", "moe"],
        help="Preview config for a performance preset",
    )
    doctor_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    doctor_parser.set_defaults(func=cmd_doctor)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
