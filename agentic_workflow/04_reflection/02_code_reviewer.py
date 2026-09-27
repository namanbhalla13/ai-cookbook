"""
Reflection-Based Code Generation Workflow
=========================================

WORKFLOW:

START
  |
  v
Generate 5 Test Cases
  |
  v
Generate Initial Code
  |
  v
Critic
  |
  |-- Reviews code against specification
  |-- Uses generated test cases
  |-- Checks expected vs actual behavior
  |-- Produces feedback
  |
  v
Decision
  |
  |-- If code is correct --------------------> END
  |
  |-- If iteration_left == 0 ----------------> END
  |
  |-- If code is incorrect and iterations remain
          |
          v
      Generate Code Again
      using previous feedback
          |
          v
        Critic
          |
          +---- repeat


IMPORTANT:

- Test cases are generated first.
- The code generator DOES NOT use the test cases.
- The code generator only uses:
    - specification
    - previous critic feedback

- The critic uses:
    - specification
    - generated code
    - generated test cases

- Feedback is accumulated across iterations.

- Maximum number of critic/refinement cycles is controlled by
  iteration_left.
"""

import operator
from typing import Annotated

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from pydantic import BaseModel, Field

from langgraph.graph import (
    START,
    END,
    StateGraph,
)

from prompt import (
    test_case_prompt,
    code_generation_prompt,
    critic_prompt,
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
# PYDANTIC MODELS
# ==============================================================


class TestCase(BaseModel):
    test_case_id: int
    test_case_input: str
    test_case_expected_output: str


class TestCaseList(BaseModel):
    test_cases: list[TestCase] = Field(
        description="Exactly 5 test cases"
    )


class TestResult(BaseModel):
    test_case_id: int
    passed: bool
    actual_output: str
    expected_output: str
    reason: str


class CriticResult(BaseModel):
    code_is_correct: bool

    feedback: list[str] = Field(
        default_factory=list
    )

    test_results: list[TestResult] = Field(
        default_factory=list
    )


# ==============================================================
# LANGGRAPH STATE
# ==============================================================


class ExtractKeyPoint(BaseModel):

    # Number of critic/refinement iterations remaining
    iteration_left: int = Field(
        default=3
    )

    # Original coding specification
    spec: str | None = Field(
        default=None,
        description=(
            "User specification used to generate "
            "and review the code."
        ),
    )

    # Test cases generated independently from the specification
    test_case: list[TestCase] = Field(
        default_factory=list
    )

    # Current generated code
    code: str | None = Field(
        default=None
    )

    # All feedback from previous critic iterations
    feedback: Annotated[
        list[str],
        operator.add,
    ] = Field(
        default_factory=list
    )

    # Most recent critic result
    critic_result: CriticResult | None = Field(
        default=None
    )


# ==============================================================
# STRUCTURED OUTPUT MODELS
# ==============================================================


test_case_model = model.with_structured_output(
    TestCaseList
)

critic_model = model.with_structured_output(
    CriticResult
)


# ==============================================================
# NODE 1 - GENERATE TEST CASES
# ==============================================================


def testcase_node(
    state: ExtractKeyPoint,
):

    test_case_chain = (
        test_case_prompt
        | test_case_model
    )

    result = test_case_chain.invoke(
        {
            "spec": state.spec
        }
    )

    return {
        "test_case": result.test_cases
    }


# ==============================================================
# NODE 2 - GENERATE / REGENERATE CODE
# ==============================================================


def generate_node(
    state: ExtractKeyPoint,
):

    # First iteration:
    # feedback will be empty.
    #
    # Later iterations:
    # feedback will contain critic comments.

    if state.feedback:
        feedback_text = "\n".join(
            f"- {feedback}"
            for feedback in state.feedback
        )
    else:
        feedback_text = "No previous feedback."


    code_chain = (
        code_generation_prompt
        | model
        | StrOutputParser()
    )


    result = code_chain.invoke(
        {
            "spec": state.spec,
            "feedback": feedback_text,
        }
    )


    return {
        "code": result
    }


# ==============================================================
# NODE 3 - CRITIC
# ==============================================================


def critic_node(
    state: ExtractKeyPoint,
):

    # Convert Pydantic TestCase objects into clean text
    # for the critic model.

    test_cases_text = "\n\n".join(
        [
            (
                f"Test Case {tc.test_case_id}\n"
                f"Input: {tc.test_case_input}\n"
                f"Expected Output: "
                f"{tc.test_case_expected_output}"
            )
            for tc in state.test_case
        ]
    )


    critic_chain = (
        critic_prompt
        | critic_model
    )


    result = critic_chain.invoke(
        {
            "spec": state.spec,
            "code": state.code,
            "test_cases": test_cases_text,
        }
    )


    return {
        "feedback": result.feedback,

        "critic_result": result,

        "iteration_left": max(
            state.iteration_left - 1,
            0,
        ),
    }


# ==============================================================
# ROUTER
# ==============================================================


def critic_router(
    state: ExtractKeyPoint,
):

    # ----------------------------------------------------------
    # CASE 1:
    # Code satisfies specification + tests
    # ----------------------------------------------------------

    if (
        state.critic_result
        and state.critic_result.code_is_correct
    ):
        return "end"


    # ----------------------------------------------------------
    # CASE 2:
    # Maximum refinement iterations reached
    # ----------------------------------------------------------

    if state.iteration_left <= 0:
        return "end"


    # ----------------------------------------------------------
    # CASE 3:
    # Problems remain and we still have iterations
    # ----------------------------------------------------------

    return "generate"


# ==============================================================
# BUILD GRAPH
# ==============================================================


graph_builder = StateGraph(
    ExtractKeyPoint
)


# ==============================================================
# ADD NODES
# ==============================================================


graph_builder.add_node(
    "testcase",
    testcase_node,
)

graph_builder.add_node(
    "generate",
    generate_node,
)

graph_builder.add_node(
    "critic",
    critic_node,
)


# ==============================================================
# GRAPH FLOW
# ==============================================================


# START
#   |
#   v
# Test Case Generator

graph_builder.add_edge(
    START,
    "testcase",
)


# Test Case Generator
#   |
#   v
# Code Generator
#
# Note:
# The generator does not read state.test_case.

graph_builder.add_edge(
    "testcase",
    "generate",
)


# Code Generator
#   |
#   v
# Critic

graph_builder.add_edge(
    "generate",
    "critic",
)


# Critic
#   |
#   |-- correct -> END
#   |
#   |-- iterations == 0 -> END
#   |
#   |-- incorrect -> Generate again

graph_builder.add_conditional_edges(
    "critic",
    critic_router,
    {
        "generate": "generate",
        "end": END,
    },
)


# ==============================================================
# COMPILE GRAPH
# ==============================================================


app = graph_builder.compile()


# ==============================================================
# TEST APPLICATION
# ==============================================================


if __name__ == "__main__":

    specification = """
Create a Python function named process_transactions(transactions).

The function processes a list of financial transactions and returns a summary.

Each transaction is represented as a dictionary with:

{
    "id": str,
    "type": str,
    "amount": int | float,
    "category": str
}

Requirements:

1. INPUT VALIDATION

- `transactions` must be a list.
- If it is not a list, raise TypeError.

Each transaction must:

- be a dictionary
- contain all four required fields:
  `id`, `type`, `amount`, and `category`

If any required field is missing, raise ValueError.

2. TRANSACTION ID

- `id` must be a non-empty string.
- Transaction IDs must be unique.
- If duplicate IDs exist, raise ValueError.

3. TRANSACTION TYPE

`type` must be either:

- "credit"
- "debit"

The comparison should be case-insensitive.

For example:

"CREDIT"
"Credit"
"credit"

must all be treated as "credit".

Any other transaction type must raise ValueError.

4. AMOUNT VALIDATION

`amount` must be an int or float.

Rules:

- amount must be greater than 0
- zero is invalid
- negative amounts are invalid
- boolean values must NOT be accepted as numbers

Invalid amounts must raise ValueError.

5. CATEGORY

`category` must be a non-empty string.

Category comparison must be case-insensitive.

For example:

"Food"
"FOOD"
"food"

must all represent the same category.

6. CALCULATE TOTALS

Calculate:

total_credit
total_debit
net_balance

where:

net_balance = total_credit - total_debit

7. CATEGORY SUMMARY

For every category, calculate:

- total_credit
- total_debit
- net_balance
- transaction_count

8. LARGEST TRANSACTION

Return the transaction with the largest amount.

If multiple transactions have the same largest amount,
return the transaction that appeared first in the input list.

9. EMPTY LIST

If `transactions` is an empty list, return:

{
    "total_credit": 0,
    "total_debit": 0,
    "net_balance": 0,
    "categories": {},
    "largest_transaction": None,
    "transaction_count": 0
}

10. OUTPUT

For a non-empty input, return a dictionary with this structure:

{
    "total_credit": ...,
    "total_debit": ...,
    "net_balance": ...,

    "categories": {
        "<normalized category>": {
            "total_credit": ...,
            "total_debit": ...,
            "net_balance": ...,
            "transaction_count": ...
        }
    },

    "largest_transaction": {
        "id": ...,
        "type": ...,
        "amount": ...,
        "category": ...
    },

    "transaction_count": ...
}

11. IMPORTANT CONSTRAINTS

- Do not modify the original input list or its dictionaries.
- Preserve the original transaction values inside
  `largest_transaction`.
- Category keys in the summary must be lowercase.
- Monetary calculations must preserve normal Python numeric behavior.
"""


    result = app.invoke(
        {
            "spec": specification
        }
    )


    # ==========================================================
    # FINAL OUTPUT
    # ==========================================================

    print("\n" + "=" * 70)
    print("FINAL CODE")
    print("=" * 70)

    print(
        result["code"]
    )


    print("\n" + "=" * 70)
    print("GENERATED TEST CASES")
    print("=" * 70)


    for test_case in result["test_case"]:

        print(
            f"\nTest Case "
            f"{test_case.test_case_id}"
        )

        print(
            f"Input: "
            f"{test_case.test_case_input}"
        )

        print(
            f"Expected: "
            f"{test_case.test_case_expected_output}"
        )


    print("\n" + "=" * 70)
    print("CRITIC RESULT")
    print("=" * 70)


    critic_result = result[
        "critic_result"
    ]


    print(
        "Code Correct:",
        critic_result.code_is_correct,
    )


    print("\nTest Results:")


    for test_result in (
        critic_result.test_results
    ):

        print(
            f"\nTest Case "
            f"{test_result.test_case_id}"
        )

        print(
            f"Passed: "
            f"{test_result.passed}"
        )

        print(
            f"Expected: "
            f"{test_result.expected_output}"
        )

        print(
            f"Actual: "
            f"{test_result.actual_output}"
        )

        print(
            f"Reason: "
            f"{test_result.reason}"
        )


    print("\n" + "=" * 70)
    print("FEEDBACK HISTORY")
    print("=" * 70)


    if result["feedback"]:

        for feedback in result["feedback"]:
            print(
                f"- {feedback}"
            )

    else:
        print(
            "No feedback. "
            "Code passed the review."
        )


    print("\nIterations Left:")

    print(
        result["iteration_left"]
    )
