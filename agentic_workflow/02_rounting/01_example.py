"""
PROBLEM STATEMENT
-----------------
Build a simple coordinator/router agent.

The coordinator receives a user request and decides which specialist
should handle it:

1. Booking Agent
   - Handles flight and hotel booking requests.

2. Information Agent
   - Handles general information questions.

3. Unclear Handler
   - Handles requests that cannot be confidently classified.


APPROACH
--------
LangChain:
    Used for:
    - OpenAI model
    - prompts
    - output parsing

LangGraph:
    Used for:
    - graph state
    - routing
    - conditional branching
    - execution flow


FLOW
----
User Request
     |
     v
Coordinator / Router LLM
     |
     |---- "booker" ----> Booking Node
     |
     |---- "info" ------> Information Node
     |
     |---- "unclear" ---> Unclear Node
                              |
                              v
                           Final Output
"""

from typing import TypedDict, Literal

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import (
    StateGraph,
    START,
    END,
)


# ==============================================================
# LOAD ENVIRONMENT
# ==============================================================

load_dotenv()


# ==============================================================
# MODEL
# ==============================================================

model = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0,
)


# ==============================================================
# LANGGRAPH STATE
# ==============================================================
#
# This is the data that moves through the graph.
#
# request:
#     Original user question
#
# decision:
#     Router decision:
#     booker / info / unclear
#
# output:
#     Final response
#

class AgentState(TypedDict):

    request: str

    decision: str

    output: str


# ==============================================================
# ROUTER PROMPT
# ==============================================================

router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a coordinator agent.

Analyze the user's request and determine which specialist
should handle it.

Routing rules:

- If the request is about booking flights or hotels,
  return: booker

- If the request is a general information question,
  return: info

- If the request is unclear or cannot be classified,
  return: unclear

Return ONLY one word:

booker
info
unclear
"""
    ),

    (
        "user",
        "{request}"
    )
])


# ==============================================================
# ROUTER CHAIN
# ==============================================================

router_chain = (
    router_prompt
    | model
    | StrOutputParser()
)


# ==============================================================
# ROUTER NODE
# ==============================================================
#
# This is LLM CALL #1.
#
# It decides where the request should go.
#

def coordinator_node(
    state: AgentState,
):

    decision = router_chain.invoke(
        {
            "request":
                state["request"]
        }
    )


    return {
        "decision":
            decision.strip().lower()
    }


# ==============================================================
# BOOKING AGENT PROMPT
# ==============================================================

booking_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a booking assistant.

Help the user with flight or hotel booking-related requests.

For this example you do not have access to a real booking system,
so explain what booking action would be required and what information
would be needed.

Keep the response concise.
"""
    ),

    (
        "user",
        "{request}"
    )
])


booking_chain = (
    booking_prompt
    | model
    | StrOutputParser()
)


# ==============================================================
# BOOKING NODE
# ==============================================================

def booking_node(
    state: AgentState,
):

    response = booking_chain.invoke(
        {
            "request":
                state["request"]
        }
    )


    return {
        "output":
            response
    }


# ==============================================================
# INFORMATION AGENT PROMPT
# ==============================================================

info_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a general information assistant.

Answer the user's general knowledge question clearly and concisely.
"""
    ),

    (
        "user",
        "{request}"
    )
])


info_chain = (
    info_prompt
    | model
    | StrOutputParser()
)


# ==============================================================
# INFORMATION NODE
# ==============================================================

def info_node(
    state: AgentState,
):

    response = info_chain.invoke(
        {
            "request":
                state["request"]
        }
    )


    return {
        "output":
            response
    }


# ==============================================================
# UNCLEAR NODE
# ==============================================================

def unclear_node(
    state: AgentState,
):

    return {
        "output": (
            "I could not determine whether this request "
            "should go to the booking or information agent. "
            "Please clarify your request."
        )
    }


# ==============================================================
# ROUTING FUNCTION
# ==============================================================
#
# LangGraph calls this AFTER coordinator_node.
#
# It checks:
#
# state["decision"]
#
# and determines which node runs next.
#

def route_request(
    state: AgentState,
) -> Literal[
    "booker",
    "info",
    "unclear",
]:

    decision = (
        state["decision"]
        .strip()
        .lower()
    )


    if decision == "booker":

        return "booker"


    if decision == "info":

        return "info"


    return "unclear"


# ==============================================================
# BUILD LANGGRAPH
# ==============================================================

builder = StateGraph(
    AgentState
)


# --------------------------------------------------------------
# Add nodes
# --------------------------------------------------------------

builder.add_node(
    "coordinator",
    coordinator_node,
)

builder.add_node(
    "booker",
    booking_node,
)

builder.add_node(
    "info",
    info_node,
)

builder.add_node(
    "unclear",
    unclear_node,
)


# --------------------------------------------------------------
# START -> coordinator
# --------------------------------------------------------------

builder.add_edge(
    START,
    "coordinator",
)


# --------------------------------------------------------------
# coordinator -> conditional routing
# --------------------------------------------------------------

builder.add_conditional_edges(
    "coordinator",

    route_request,

    {
        "booker":
            "booker",

        "info":
            "info",

        "unclear":
            "unclear",
    },
)


# --------------------------------------------------------------
# Specialist nodes -> END
# --------------------------------------------------------------

builder.add_edge(
    "booker",
    END,
)

builder.add_edge(
    "info",
    END,
)

builder.add_edge(
    "unclear",
    END,
)


# ==============================================================
# COMPILE GRAPH
# ==============================================================

coordinator_agent = (
    builder.compile()
)


# ==============================================================
# TEST
# ==============================================================

def main():

    print(
        "\n--- BOOKING REQUEST ---"
    )

    result_a = (
        coordinator_agent.invoke(
            {
                "request":
                    "Book me a flight to London."
            }
        )
    )

    print(
        "Decision:",
        result_a["decision"]
    )

    print(
        "Response:",
        result_a["output"]
    )


    print(
        "\n--- INFORMATION REQUEST ---"
    )

    result_b = (
        coordinator_agent.invoke(
            {
                "request":
                    "What is the capital of Italy?"
            }
        )
    )

    print(
        "Decision:",
        result_b["decision"]
    )

    print(
        "Response:",
        result_b["output"]
    )


    print(
        "\n--- UNCLEAR REQUEST ---"
    )

    result_c = (
        coordinator_agent.invoke(
            {
                "request":
                    "asdf something random xyz"
            }
        )
    )

    print(
        "Decision:",
        result_c["decision"]
    )

    print(
        "Response:",
        result_c["output"]
    )


if __name__ == "__main__":

    main()
