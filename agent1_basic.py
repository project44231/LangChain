import os

from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()  # reads .env in the project root
openai_key = os.getenv("openai_key")
if not openai_key:
    raise RuntimeError("openai_key missing from .env")

os.environ["OPENAI_API_KEY"] = openai_key  # required by LangChain

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
user_question = input("Enter your question: ")
result = agent.invoke({"messages": [{"role": "user", "content": user_question}]})

messages = result["messages"]
assistant_msg = next(
    (msg for msg in reversed(messages) if msg.type == "ai"),
    None,
)
if assistant_msg is None:
    raise RuntimeError("No assistant response found")

content = assistant_msg.content
if isinstance(content, list):
    text = "\n".join(
        block["text"] for block in content if block.get("type") == "text"
    )
else:
    text = content

print(text)