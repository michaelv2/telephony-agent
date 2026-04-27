"""CLI harness for testing CallAgent conversations without the voice pipeline."""

import asyncio
import sys

import config
from agent import CallAgent


async def main():
    context = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    agent = CallAgent(context=context)

    print(f"Agent prompt: {config.AGENT_PROMPT[:80]}...")
    if context:
        print(f"Call context: {context}")
    print("Type messages as the caller. Ctrl+C to quit.\n")

    # Greeting
    print("Agent: ", end="", flush=True)
    async for sentence in agent.greeting_stream():
        print(sentence, end=" ", flush=True)
    print("\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue

        print("Agent: ", end="", flush=True)
        async for sentence in agent.respond_stream(user_input):
            print(sentence, end=" ", flush=True)
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
