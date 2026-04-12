"""Command-line interface for Miniforge."""

import argparse
import asyncio
import sys
from pathlib import Path

from miniforge.utils.config import M7Config, create_default_config_file
from miniforge.models.registry import get_registry


def cmd_chat(args):
    """Run interactive chat."""
    from miniforge import Miniforge

    async def interactive():
        config = M7Config.from_yaml(args.config) if args.config else M7Config()

        print("Loading MiniMax M2.7...")
        model = await Miniforge.from_pretrained(
            args.model or "MiniMaxAI/MiniMax-M2.7",
            quantization=args.quantization,
            config=config,
        )
        print("Model loaded! Type 'quit' to exit.\n")

        history = []

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                response = await model.chat(
                    message=user_input,
                    history=history,
                    stream=args.stream,
                )

                if args.stream:
                    print("Assistant: ", end="", flush=True)
                    async for token in response:
                        print(token, end="", flush=True)
                    print("\n")
                else:
                    print(f"Assistant: {response}\n")

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
    print("Server mode not yet implemented.")
    print("Use examples/server_api.py for now.")


def cmd_download(args):
    """Download model."""
    from miniforge.models.registry import get_registry

    registry = get_registry()

    print(f"Downloading {args.model}...")

    try:
        path = registry.download_hf_model(args.model)
        print(f"Downloaded to: {path}")

        # Check for GGUF
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


def cmd_config(args):
    """Manage configuration."""
    if args.create:
        path = create_default_config_file()
        print(f"Created config at: {path}")
    else:
        config = M7Config()
        print("Current configuration:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="miniforge",
        description="Miniforge: High-performance inference for GMKtech M7",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument("--model", "-m", help="Model ID or path")
    chat_parser.add_argument("--quantization", "-q", default="Q4_K_M", help="Quantization type")
    chat_parser.add_argument("--config", "-c", help="Config file path")
    chat_parser.add_argument("--stream", "-s", action="store_true", help="Stream output")
    chat_parser.set_defaults(func=cmd_chat)

    # Server command
    server_parser = subparsers.add_parser("serve", help="Run API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    server_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind")
    server_parser.add_argument("--model", "-m", help="Model ID or path")
    server_parser.set_defaults(func=cmd_server)

    # Download command
    download_parser = subparsers.add_parser("download", help="Download model")
    download_parser.add_argument("model", help="Model ID to download")
    download_parser.set_defaults(func=cmd_download)

    # List command
    list_parser = subparsers.add_parser("list", help="List cached models")
    list_parser.set_defaults(func=cmd_list)

    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("--create", action="store_true", help="Create default config")
    config_parser.set_defaults(func=cmd_config)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
