import os

from dataclasses import dataclass
from dotenv import load_dotenv

# LangChain core imports for agent creation and orchestration
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
# Load environment variables from .env file in the project root directory
# This allows secure storage of API keys without hardcoding them
load_dotenv()  # reads .env in the project root

# Retrieve the OpenAI API key from environment variables
# The key is stored in .env file as "openai_key=your_key_here"
openai_key = os.getenv("openai_key")
if not openai_key:
    raise RuntimeError("openai_key missing from .env")

# Set the OPENAI_API_KEY environment variable that LangChain/OpenAI SDK expects
# This is required for authentication when making API calls to OpenAI
os.environ["OPENAI_API_KEY"] = openai_key  # required by LangChain

# ============================================================================
# SYSTEM PROMPT CONFIGURATION
# ============================================================================
# The system prompt defines the agent's personality, behavior, and capabilities
# This prompt is sent to the LLM to establish its role and available tools
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

# ============================================================================
# CONTEXT SCHEMA DEFINITION
# ============================================================================
# @dataclass decorator creates a class with automatically generated __init__, 
# __repr__, and other special methods based on the defined fields
# This Context class defines the runtime context that can be passed to tools
# It allows tools to access user-specific information during execution
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str  # Unique identifier for the user making the request

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================
# Tools are functions that the agent can call to perform actions or retrieve data
# The @tool decorator registers these functions as callable tools for the agent

@tool
def get_weather_for_location(city: str) -> str:
    """
    Tool to retrieve weather information for a specific city.
    
    Args:
        city: Name of the city to get weather for
        
    Returns:
        Weather information string (currently a mock response)
    """
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """
    Tool to retrieve user's location based on their user ID.
    
    This tool demonstrates how to access runtime context within a tool.
    The ToolRuntime[Context] parameter provides access to the Context object
    that was passed when invoking the agent.
    
    Args:
        runtime: ToolRuntime object containing the Context with user_id
        
    Returns:
        Location string based on user_id mapping
    """
    user_id = runtime.context.user_id
    # Simple mapping: user_id "1" maps to "Florida", others map to "SF"
    return "Florida" if user_id == "1" else "SF"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Initialize the chat model that will power the agent
# init_chat_model automatically detects the model provider (OpenAI, Anthropic, etc.)
# based on the model name string
model = init_chat_model(
    "gpt-5-nano",  # Model identifier - OpenAI's GPT-5 Nano model
    temperature=0  # Temperature controls randomness: 0 = deterministic, higher = more creative
)

# ============================================================================
# RESPONSE FORMAT SCHEMA
# ============================================================================
# Define a structured output format using a dataclass
# This ensures the agent's responses follow a specific schema
# ToolStrategy enforces this format when the agent generates responses
@dataclass
class ResponseFormat:
    """
    Response schema for the agent's structured output.
    
    This defines the exact structure the agent must return:
    - punny_response: Always required, contains the agent's pun-filled response
    - weather_conditions: Optional field for weather information
    """
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None  # Optional field (can be None)

# ============================================================================
# MEMORY/CONTEXT MANAGEMENT
# ============================================================================
# InMemorySaver provides conversation memory/checkpointing functionality
# This allows the agent to remember previous interactions in a conversation
# The memory is stored in RAM (not persistent across restarts)
checkpointer = InMemorySaver()

# ============================================================================
# AGENT CREATION
# ============================================================================
# create_agent builds a LangGraph agent with the specified configuration
# The agent combines:
#   - LLM model for reasoning and text generation
#   - Tools for executing actions
#   - System prompt for behavior definition
#   - Context schema for runtime data
#   - Response format for structured outputs
#   - Checkpointer for conversation memory
agent = create_agent(
    model=model,  # The initialized chat model (GPT-5 Nano)
    system_prompt=SYSTEM_PROMPT,  # Defines agent's role and behavior
    tools=[get_user_location, get_weather_for_location],  # Available tools
    context_schema=Context,  # Schema for runtime context data
    response_format=ToolStrategy(ResponseFormat),  # Enforce structured output
    checkpointer=checkpointer  # Enable conversation memory
)

# ============================================================================
# AGENT EXECUTION - FIRST INTERACTION
# ============================================================================
# thread_id is a unique identifier for a conversation thread
# Using the same thread_id allows the agent to maintain conversation context
# across multiple invocations (conversation memory)
config = {"configurable": {"thread_id": "1"}}

# Invoke the agent with a user message
# The agent will:
#   1. Process the user's question
#   2. Decide which tools to call (if any)
#   3. Execute tools with the provided context
#   4. Generate a structured response following ResponseFormat schema
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,  # Use thread_id "1" for this conversation
    context=Context(user_id="1")  # Pass user context to tools
)

# Print the structured response from the agent
# response['structured_response'] contains the ResponseFormat object
print(response['structured_response'])
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# ============================================================================
# AGENT EXECUTION - CONTINUED CONVERSATION
# ============================================================================
# Using the same thread_id ("1") allows the agent to remember the previous
# conversation context. The checkpointer maintains the conversation history
# so the agent can reference earlier messages.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,  # Same thread_id maintains conversation context
    context=Context(user_id="1")  # Same user context
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )