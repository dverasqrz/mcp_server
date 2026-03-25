from agents import Agent, Runner, function_tool

from dotenv import load_dotenv

load_dotenv("../.env")


@function_tool
def multiply(x: float, y: float) -> float:
    """
    Multiply two numbers
    """
    return x * y


agent = Agent(
    name="Math Agent",
    instructions="Always use your tools to solve math problems.",
    tools=[multiply],
)

response = Runner.run_sync(agent, "What is 6.2 times 3.5?")
print(response.final_output)
