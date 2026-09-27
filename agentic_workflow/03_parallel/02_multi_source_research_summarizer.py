from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langgraph.graph import (
    START,
    END,
    StateGraph,
)


# ==============================================================
# LOAD ENVIRONMENT VARIABLES
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
# STATE
# ==============================================================

class ExtractKeyPoint(BaseModel):

    # Original request from user
    topic_original: str | None = Field(
        default=None
    )

    # Extracted topic
    topic: str | None = Field(
        default=None
    )

    # Specialist-agent outputs
    pros: str | None = Field(
        default=None
    )

    cons: str | None = Field(
        default=None
    )

    recent_development: str | None = Field(
        default=None
    )

    # Final synthesized output
    output: str | None = Field(
        default=None
    )


# ==============================================================
# TOPIC EXTRACTION PROMPT
# ==============================================================

topic_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a topic extraction assistant.

Your task is to identify the main topic that the user wants to
analyze or research.

Rules:

1. Extract only the primary topic.
2. Remove conversational wording and instructions.
3. Preserve important qualifiers that define the topic.
4. Do not answer or analyze the topic.
5. Return only the topic name.

Examples:

User:
"Tell me the pros and cons of electric vehicles."

Output:
Electric Vehicles


User:
"I want to understand recent developments in generative AI."

Output:
Generative AI


User:
"Can you analyze remote work for software companies?"

Output:
Remote Work in Software Companies
"""
        ),
        (
            "user",
            """
User Request:

{user_request}
"""
        ),
    ]
)


# ==============================================================
# PROS PROMPT
# ==============================================================

prompt_pro = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a subject-matter expert responsible for analyzing the
positive aspects, advantages, and potential benefits of a given
topic.

Your task is to provide a thoughtful and well-structured analysis
of the PROS of the topic.

When analyzing the topic:

1. Identify the major advantages and benefits.

2. Explain WHY each advantage is important.

3. Describe the potential practical impact of each advantage.

4. Consider different perspectives where relevant, such as:
   - individuals
   - businesses
   - society
   - technology
   - economy
   - environment

5. Provide concrete examples when they help explain an advantage.

6. Avoid repeating similar points.

7. Do not discuss disadvantages unless necessary to clarify an
   advantage.

8. Do not invent facts, statistics, or evidence.

9. Keep the analysis clear, concise, and logically structured.

Return the most important pros as separate points, with a short
explanation for each.
"""
        ),
        (
            "user",
            """
Analyze the following topic and explain its main advantages
and benefits:

Topic:
{topic}
"""
        ),
    ]
)


# ==============================================================
# CONS PROMPT
# ==============================================================

prompt_con = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a subject-matter expert responsible for analyzing the
negative aspects, disadvantages, limitations, and risks of a
given topic.

Your task is to provide a thoughtful and well-structured analysis
of the CONS of the topic.

When analyzing the topic:

1. Identify the major disadvantages and limitations.

2. Explain WHY each disadvantage is important.

3. Describe the potential practical impact of each disadvantage.

4. Consider different perspectives where relevant, such as:
   - individuals
   - businesses
   - society
   - technology
   - economy
   - environment

5. Identify important risks, challenges, or barriers.

6. Provide concrete examples when they help explain a limitation.

7. Avoid repeating similar points.

8. Do not focus on advantages unless necessary to explain a
   limitation.

9. Do not invent facts, statistics, or evidence.

10. Keep the analysis clear, concise, and logically structured.

Return the most important cons as separate points, with a short
explanation for each.
"""
        ),
        (
            "user",
            """
Analyze the following topic and explain its main disadvantages,
limitations, risks, and challenges:

Topic:
{topic}
"""
        ),
    ]
)


# ==============================================================
# RECENT DEVELOPMENTS PROMPT
# ==============================================================

prompt_recent_developments = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a research analyst responsible for identifying and
explaining recent developments related to a given topic.

Your task is to provide a clear and structured overview of recent
developments based only on the information available to you.

When analyzing recent developments:

1. Identify important recent changes, events, innovations,
   or trends.

2. Explain what changed or developed.

3. Explain why the development is important.

4. Describe its potential impact.

5. Consider developments across relevant areas such as:
   - technology
   - business and industry
   - research
   - regulation and policy
   - market trends
   - adoption and implementation

6. Include dates or time periods when you are confident they
   are accurate.

7. Prioritize meaningful developments rather than minor updates.

8. Do not invent events, dates, statistics, or announcements.

9. Clearly indicate when current information is unavailable
   or uncertain.

10. Keep the analysis concise and logically structured.

Return the most important developments as separate points with a
short explanation of their significance.
"""
        ),
        (
            "user",
            """
Identify and explain the recent important developments related
to the following topic:

Topic:
{topic}
"""
        ),
    ]
)


# ==============================================================
# SUMMARY PROMPT
# ==============================================================

summary_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a synthesis and summary agent.

You receive outputs from multiple specialist agents about the
same topic.

Your task is to combine those outputs into one clear, coherent,
balanced summary.

Use ONLY the information supplied by the specialist agents.

Rules:

1. Do not add outside knowledge, assumptions, or new facts.

2. Remove repetition between the specialist-agent outputs.

3. Combine related or overlapping ideas.

4. Preserve important facts, dates, numbers, names, caveats,
   and uncertainties.

5. If specialist agents provide conflicting information,
   explicitly mention the conflict.

6. Do not merely concatenate the specialist responses.

7. Synthesize them into one natural explanation.

8. Balance advantages and limitations.

9. Give appropriate importance to significant recent
   developments.

10. If one agent provides little or no useful information,
    do not invent information to fill the gap.

11. Prioritize important insights over minor details.

Return only the final synthesized summary.
"""
        ),
        (
            "user",
            """
TOPIC:

{topic}


PROS:

{pros}


CONS:

{cons}


RECENT DEVELOPMENTS:

{recent_development}


Create the final response using this structure:

## Overview

Provide a concise introduction based on the information supplied
by the specialist agents.


## Key Advantages

Summarize the most important advantages.


## Key Limitations

Summarize the most important disadvantages, risks, and
limitations.


## Recent Developments

Summarize the most important recent developments.


## Overall Summary

Provide a concise synthesis connecting the advantages,
limitations, and recent developments.
"""
        ),
    ]
)


# ==============================================================
# CHAINS
# ==============================================================

topic_extraction_chain = (
    topic_extraction_prompt
    | model
    | StrOutputParser()
)

pros_chain = (
    prompt_pro
    | model
    | StrOutputParser()
)

cons_chain = (
    prompt_con
    | model
    | StrOutputParser()
)

recent_development_chain = (
    prompt_recent_developments
    | model
    | StrOutputParser()
)

summary_chain = (
    summary_agent_prompt
    | model
    | StrOutputParser()
)


# ==============================================================
# NODE 1 — EXTRACT TOPIC
# ==============================================================

def extract_topic_node(
    state: ExtractKeyPoint,
):

    result = topic_extraction_chain.invoke(
        {
            "user_request": state.topic_original
        }
    )

    return {
        "topic": result.strip()
    }


# ==============================================================
# NODE 2 — GENERATE PROS
# ==============================================================

def generate_pros_node(
    state: ExtractKeyPoint,
):

    result = pros_chain.invoke(
        {
            "topic": state.topic
        }
    )

    return {
        "pros": result
    }


# ==============================================================
# NODE 3 — GENERATE CONS
# ==============================================================

def generate_cons_node(
    state: ExtractKeyPoint,
):

    result = cons_chain.invoke(
        {
            "topic": state.topic
        }
    )

    return {
        "cons": result
    }


# ==============================================================
# NODE 4 — RECENT DEVELOPMENTS
# ==============================================================

def research_recent_developments_node(
    state: ExtractKeyPoint,
):

    result = recent_development_chain.invoke(
        {
            "topic": state.topic
        }
    )

    return {
        "recent_development": result
    }


# ==============================================================
# NODE 5 — FINAL SUMMARY
# ==============================================================

def synthesize_summary_node(
    state: ExtractKeyPoint,
):

    result = summary_chain.invoke(
        {
            "topic": state.topic,
            "pros": state.pros,
            "cons": state.cons,
            "recent_development": state.recent_development,
        }
    )

    return {
        "output": result
    }


# ==============================================================
# BUILD LANGGRAPH
# ==============================================================

graph_builder = StateGraph(
    ExtractKeyPoint
)


# ==============================================================
# ADD NODES
# ==============================================================

graph_builder.add_node(
    "extract_topic",
    extract_topic_node,
)

graph_builder.add_node(
    "generate_pros",
    generate_pros_node,
)

graph_builder.add_node(
    "generate_cons",
    generate_cons_node,
)

graph_builder.add_node(
    "recent_developments",
    research_recent_developments_node,
)

graph_builder.add_node(
    "summary",
    synthesize_summary_node,
)


# ==============================================================
# GRAPH FLOW
# ==============================================================

graph_builder.add_edge(
    START,
    "extract_topic",
)


# --------------------------------------------------------------
# FAN OUT
#
# After extracting the topic:
#
#               extract topic
#                    |
#        ---------------------------
#        |            |            |
#       pros         cons       recent
#
# --------------------------------------------------------------

graph_builder.add_edge(
    "extract_topic",
    "generate_pros",
)

graph_builder.add_edge(
    "extract_topic",
    "generate_cons",
)

graph_builder.add_edge(
    "extract_topic",
    "recent_developments",
)


# --------------------------------------------------------------
# FAN IN
#
# Wait until ALL three specialist agents have completed
# before running the summary agent.
# --------------------------------------------------------------

graph_builder.add_edge(
    [
        "generate_pros",
        "generate_cons",
        "recent_developments",
    ],
    "summary",
)


# --------------------------------------------------------------
# END
# --------------------------------------------------------------

graph_builder.add_edge(
    "summary",
    END,
)


# ==============================================================
# COMPILE GRAPH
# ==============================================================

app = graph_builder.compile()


# ==============================================================
# TEST USE CASE 1
# ==============================================================

def test_use_case_1():

    print("\n")
    print("=" * 80)
    print("TEST CASE 1")
    print("=" * 80)

    user_request = (
        "I want to understand the pros, cons, "
        "and recent developments in electric vehicles."
    )

    print("\nUSER REQUEST:")
    print(user_request)

    result = app.invoke(
        {
            "topic_original": user_request
        }
    )

    print("\nEXTRACTED TOPIC:")
    print(result["topic"])

    print("\nFINAL OUTPUT:")
    print(result["output"])


# ==============================================================
# TEST USE CASE 2
# ==============================================================

def test_use_case_2():

    print("\n")
    print("=" * 80)
    print("TEST CASE 2")
    print("=" * 80)

    user_request = (
        "Analyze generative AI and tell me its major benefits, "
        "limitations, and recent developments."
    )

    print("\nUSER REQUEST:")
    print(user_request)

    result = app.invoke(
        {
            "topic_original": user_request
        }
    )

    print("\nEXTRACTED TOPIC:")
    print(result["topic"])

    print("\nFINAL OUTPUT:")
    print(result["output"])


# ==============================================================
# RUN TESTS
# ==============================================================

if __name__ == "__main__":

    test_use_case_1()

    test_use_case_2()
