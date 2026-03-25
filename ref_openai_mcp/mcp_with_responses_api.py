# %%
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../.env")

client = OpenAI()

# %%
resp = client.responses.create(model="gpt-4.1", tools=[], input="Say Hello!")

print(resp.model_dump_json(indent=2))
# %%

prompt = """Take a look at deepwiki and figure out What transport
            protocols are supported in the 2025-03-26 version of the MCP spec?
            Using the modelcontextprotocol/python-sdk repo"""

resp = client.responses.create(
    model="gpt-4.1",
    tools=[
        {
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
        }
    ],
    input=prompt,
)

print(resp.model_dump_json(indent=2))

# %%
request_id = resp.output[-1].id

resp = client.responses.create(
    model="gpt-4.1",
    tools=[
        {
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
        }
    ],
    previous_response_id=resp.id,
    input=[
        {
            "type": "mcp_approval_response",
            "approve": True,
            "approval_request_id": request_id,
        }
    ],
)

print(resp.model_dump_json(indent=2))

#%% 


prompt = """Take a look at deepwiki and figure out What transport
            protocols are supported in the 2025-03-26 version of the MCP spec?
            Using the modelcontextprotocol/python-sdk repo"""

resp = client.responses.create(
    model="gpt-4.1",
    tools=[
        {
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
            "require_approval": "never",
            "allowed_tools": ["ask_question"]
        }
    ],
    input=prompt,
)

print(resp.output_text)
#%%
