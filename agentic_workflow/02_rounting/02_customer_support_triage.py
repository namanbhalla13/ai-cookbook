"""
PROBLEM STATEMENT
-----------------
Build a customer-support routing workflow.

The system should:

1. Receive a customer request.
2. Use an LLM triage agent to classify the request as:
   - billing
   - technical
   - refund
   - general

3. If the request is unclear:
   - do not route it
   - return a follow-up question

4. If the request is clear:
   - route it to the correct specialist
   - let that specialist generate the final response


APPROACH
--------
LangChain:
    - OpenAI model
    - prompts
    - output parsers

LangGraph:
    - shared workflow state
    - nodes
    - conditional routing
    - execution flow


FLOW
----
Customer Request
       |
       v
    TRIAGE
       |
       |---- billing ------> Billing Specialist
       |
       |---- technical ----> Technical Specialist
       |
       |---- refund -------> Refund Specialist
       |
       |---- general ------> General Specialist
       |
       |---- unclear ------> Follow-up
                                |
                                v
                               END
"""

from typing import TypedDict, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from prompt import (
    triage_prompt,
    billing_prompt,
    technical_prompt,
    refund_prompt,
    general_prompt,
)


# ==============================================================
# ENVIRONMENT
# ==============================================================

load_dotenv()


# ==============================================================
# MODEL
# ==============================================================

model = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0.3,
)


# ==============================================================
# TRIAGE STRUCTURED OUTPUT
# ==============================================================

class TriageResult(BaseModel):

    category: Literal[
        "billing",
        "technical",
        "refund",
        "general"
    ] | None = Field(
        default=None,
        description=(
            "Routing category. Return None if the request "
            "cannot be confidently routed."
        )
    )

    follow_up_required: bool = Field(
        description=(
            "True if more information is required "
            "before routing."
        )
    )

    follow_up_question: str | None = Field(
        default=None,
        description=(
            "Follow-up question when routing is unclear. "
            "Otherwise return None."
        )
    )


# ==============================================================
# LANGGRAPH STATE
# ==============================================================
#
# This object moves through the entire graph.
#
# customer_request:
#     Original customer request
#
# category:
#     Result from triage
#
# follow_up_required:
#     Whether clarification is required
#
# follow_up_question:
#     Question generated when routing is unclear
#
# response:
#     Final specialist response
#

class SupportState(TypedDict):

    customer_request: str

    category: str | None

    follow_up_required: bool

    follow_up_question: str | None

    response: str | None


# ==============================================================
# TRIAGE PARSER
# ==============================================================

triage_parser = PydanticOutputParser(
    pydantic_object=TriageResult
)


triage_prompt_with_format = triage_prompt.partial(
    format_instructions=
        triage_parser.get_format_instructions()
)


# ==============================================================
# TRIAGE CHAIN
# ==============================================================

triage_chain = (
    triage_prompt_with_format
    | model
    | triage_parser
)


# ==============================================================
# SPECIALIST CHAINS
# ==============================================================

billing_chain = (
    billing_prompt
    | model
    | StrOutputParser()
)


technical_chain = (
    technical_prompt
    | model
    | StrOutputParser()
)


refund_chain = (
    refund_prompt
    | model
    | StrOutputParser()
)


general_chain = (
    general_prompt
    | model
    | StrOutputParser()
)


# ==============================================================
# TRIAGE NODE
# ==============================================================

def triage_node(
    state: SupportState
):

    result = triage_chain.invoke({
        "customer_request":
            state["customer_request"]
    })


    return {
        "category":
            result.category,

        "follow_up_required":
            result.follow_up_required,

        "follow_up_question":
            result.follow_up_question,
    }


# ==============================================================
# BILLING NODE
# ==============================================================

def billing_node(
    state: SupportState
):

    response = billing_chain.invoke({
        "customer_request":
            state["customer_request"]
    })


    return {
        "response":
            response
    }


# ==============================================================
# TECHNICAL NODE
# ==============================================================

def technical_node(
    state: SupportState
):

    response = technical_chain.invoke({
        "customer_request":
            state["customer_request"]
    })


    return {
        "response":
            response
    }


# ==============================================================
# REFUND NODE
# ==============================================================

def refund_node(
    state: SupportState
):

    response = refund_chain.invoke({
        "customer_request":
            state["customer_request"]
    })


    return {
        "response":
            response
    }


# ==============================================================
# GENERAL NODE
# ==============================================================

def general_node(
    state: SupportState
):

    response = general_chain.invoke({
        "customer_request":
            state["customer_request"]
    })


    return {
        "response":
            response
    }


# ==============================================================
# FOLLOW-UP NODE
# ==============================================================

def follow_up_node(
    state: SupportState
):

    return {
        "response":
            state["follow_up_question"]
    }


# ==============================================================
# ROUTER
# ==============================================================
#
# This function does NOT call an LLM.
#
# It simply looks at the result produced by triage_node
# and tells LangGraph which node should run next.
#

def route_after_triage(
    state: SupportState
) -> Literal[
    "billing",
    "technical",
    "refund",
    "general",
    "follow_up",
]:

    if state["follow_up_required"]:

        return "follow_up"


    if state["category"] == "billing":

        return "billing"


    if state["category"] == "technical":

        return "technical"


    if state["category"] == "refund":

        return "refund"


    if state["category"] == "general":

        return "general"


    return "follow_up"


# ==============================================================
# BUILD LANGGRAPH
# ==============================================================

graph_builder = StateGraph(
    SupportState
)


# Add graph nodes

graph_builder.add_node(
    "triage",
    triage_node,
)

graph_builder.add_node(
    "billing",
    billing_node,
)

graph_builder.add_node(
    "technical",
    technical_node,
)

graph_builder.add_node(
    "refund",
    refund_node,
)

graph_builder.add_node(
    "general",
    general_node,
)

graph_builder.add_node(
    "follow_up",
    follow_up_node,
)


# ==============================================================
# GRAPH FLOW
# ==============================================================

graph_builder.add_edge(
    START,
    "triage",
)


graph_builder.add_conditional_edges(
    "triage",

    route_after_triage,

    {
        "billing":
            "billing",

        "technical":
            "technical",

        "refund":
            "refund",

        "general":
            "general",

        "follow_up":
            "follow_up",
    },
)


# Every specialist ends the workflow

graph_builder.add_edge(
    "billing",
    END,
)

graph_builder.add_edge(
    "technical",
    END,
)

graph_builder.add_edge(
    "refund",
    END,
)

graph_builder.add_edge(
    "general",
    END,
)

graph_builder.add_edge(
    "follow_up",
    END,
)


# ==============================================================
# COMPILE GRAPH
# ==============================================================

support_graph = (
    graph_builder.compile()
)


# ==============================================================
# TEST
# ==============================================================

test_cases = [

    "I was charged twice for my monthly subscription.",

    "The app crashes every time I upload a PDF.",

    "I cancelled my subscription and want my money back.",

    "What features are included in the premium plan?",

    "I have a problem with my account.",
]


for request in test_cases:

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CUSTOMER:"
    )

    print(
        request
    )


    result = support_graph.invoke({

        "customer_request":
            request,

        "category":
            None,

        "follow_up_required":
            False,

        "follow_up_question":
            None,

        "response":
            None,
    })


    print(
        "\nCATEGORY:"
    )

    print(
        result["category"]
    )


    print(
        "\nSUPPORT RESPONSE:"
    )

    print(
        result["response"]
    )
