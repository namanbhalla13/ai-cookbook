"""
PROBLEM STATEMENT
-----------------
Given one topic, run three independent LLM tasks in parallel:

1. Generate a concise summary
2. Generate three interesting questions
3. Extract 5-10 key terms

Then combine all three outputs into one final synthesis response.

APPROACH
--------
Use LangChain Runnables only:

Topic
  |
  v
RunnableParallel
  |---- summary_chain
  |---- questions_chain
  |---- key_terms_chain
  |---- original topic
  |
  v
synthesis_chain
  |
  v
Final Answer
"""

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)


load_dotenv()


# ==============================================================
# MODEL
# ==============================================================

model = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0.3,
)


# ==============================================================
# SUMMARY CHAIN
# ==============================================================

summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Summarize the following topic concisely.

Focus on:
- main idea
- important background
- important developments
"""
    ),
    (
        "user",
        "{topic}"
    )
])


summary_chain = (
    summary_prompt
    | model
    | StrOutputParser()
)


# ==============================================================
# QUESTIONS CHAIN
# ==============================================================

questions_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Generate three interesting questions about the following topic.

The questions should help someone explore the topic more deeply.
"""
    ),
    (
        "user",
        "{topic}"
    )
])


questions_chain = (
    questions_prompt
    | model
    | StrOutputParser()
)


# ==============================================================
# KEY TERMS CHAIN
# ==============================================================

key_terms_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Identify 5-10 important key terms from the following topic.

Return them as a comma-separated list.
"""
    ),
    (
        "user",
        "{topic}"
    )
])


key_terms_chain = (
    key_terms_prompt
    | model
    | StrOutputParser()
)


# ==============================================================
# PARALLEL STEP
# ==============================================================
#
# All three chains receive the same input:
#
# {
#     "topic": "The history of space exploration"
# }
#
# and execute independently.
#
# Output:
#
# {
#     "summary": "...",
#     "questions": "...",
#     "key_terms": "...",
#     "topic": "..."
# }
#
# ==============================================================

parallel_chain = RunnableParallel(

    summary=summary_chain,

    questions=questions_chain,

    key_terms=key_terms_chain,

    topic=RunnablePassthrough()
)


# ==============================================================
# SYNTHESIS PROMPT
# ==============================================================

synthesis_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a research synthesis assistant.

Using the information generated below, create one clear
and comprehensive explanation.

Summary:
{summary}

Related Questions:
{questions}

Key Terms:
{key_terms}

Combine the information naturally.
"""
    ),
    (
        "user",
        """
Original Topic:

{topic}
"""
    )
])


# ==============================================================
# SYNTHESIS CHAIN
# ==============================================================

synthesis_chain = (
    synthesis_prompt
    | model
    | StrOutputParser()
)


# ==============================================================
# COMPLETE PIPELINE
# ==============================================================

full_parallel_chain = (
    parallel_chain
    | synthesis_chain
)


# ==============================================================
# RUN
# ==============================================================

result = full_parallel_chain.invoke({
    "topic": "The history of space exploration"
})


print(result)
