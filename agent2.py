"""
An AI agent with playwright mcp server, and Gemini-2.0-flash."""

import os
import asyncio
from dotenv import load_dotenv
from agents import (
    Runner,
    Agent,
    OpenAIChatCompletionsModel,
    set_default_openai_client,
    set_tracing_disabled
)
from agents.mcp import MCPServerStdio
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BRIGHT_DATA_API_TOKEN = os.getenv("BRIGHT_DATA_API_TOKEN")
BRIGHT_DATA_BROWSER_AUTH = os.getenv("BRIGHT_DATA_BROWSER_AUTH")


async def create_mcp_ai_agent(mcp_server):
    """
    Create an AI agent configured to use the MCP server with Gemini-2.0-flash model."""
    gemini_client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    set_default_openai_client(gemini_client)
    set_tracing_disabled(True)

    # create an agent configured to use the MCP server
    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant.",
        model=OpenAIChatCompletionsModel(
            model="gemini-2.0-flash",
            openai_client=gemini_client,
        ),
        mcp_servers=[mcp_server]
    )
    return agent


async def run():
    """
    Main function to run the AI agent with MCP server.
    It reads user input from stdin, processes it through the agent,
    and prints the output."""

    # AI agent logic goes here
    async with MCPServerStdio(
        name="Bright Data Web data MCP server, via npx",
        params={
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest", "--output-dir", "./", "--cdp-endpoint", BRIGHT_DATA_BROWSER_AUTH],
        },
        # To avoid timeout issues
        client_session_timeout_seconds=180
        ) as server:
        agent = await create_mcp_ai_agent(server)

        while True:
            # Read user input from stdin
            request = input("Your request -> ")

            # Exit condition
            if request.lower() == "exit":
                print("Exiting the agent...")
                break

            # Run the request through the agent
            print("Running request...")
            output = await Runner.run(agent, input=request)

            # Print the output from the agent
            print(f"Output -> \n{output}\n\n")

if __name__ == "__main__":
    asyncio.run(run())
