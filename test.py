import asyncio
from miniforge import Miniforge

async def main():
    # Load model (auto-downloads GGUF if available)
    model = await Miniforge.from_pretrained(
        "MiniMaxAI/MiniMax-M2.7",
        quantization="BF16",
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