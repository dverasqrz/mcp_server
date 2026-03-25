import asyncio
from agents import Agent, Runner, function_tool
from agents.mcp.server import MCPServerSse
from dotenv import load_dotenv
from agents.model_settings import ModelSettings

load_dotenv("../.env")


async def main():
    async with MCPServerSse(
        name="DeepWiki MCP Server",
        params={
            "url": "https://mcp.deepwiki.com/sse",
        },
        client_session_timeout_seconds=30
    ) as mcp_server:
        agent = Agent(
            name="DeepWiki Agent",
            instructions="Use the tools to answer the questions.",
            mcp_servers=[mcp_server],
        )

        prompt = """Take a look at deepwiki and figure out What transport
                    protocols are supported in the 2025-03-26 version of the MCP spec?
                    Using the modelcontextprotocol/python-sdk repo"""


        response = await Runner.run(agent, prompt)
        print(response.final_output)

asyncio.run(main())
