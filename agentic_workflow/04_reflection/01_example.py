import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ==============================================================
# CONFIGURATION
# ==============================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not found in .env file. Please add it."
    )


llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0.1,
)


# ==============================================================
# REFLECTION LOOP
# ==============================================================

def run_reflection_loop():
    """
    Demonstrates a multi-step AI reflection loop to progressively
    improve a Python function.
    """

    task_prompt = """
Your task is to create a Python function named `calculate_factorial`.

The function should:

1. Accept a single integer `n` as input.
2. Calculate its factorial (n!).
3. Include a clear docstring explaining what the function does.
4. Handle the edge case where factorial of 0 is 1.
5. Raise a ValueError if the input is a negative number.

Return only the Python code for the function.
"""

    max_iterations = 3
    current_code = ""

    message_history = [
        HumanMessage(content=task_prompt)
    ]

    for i in range(max_iterations):

        print(
            "\n"
            + "=" * 25
            + f" REFLECTION LOOP: ITERATION {i + 1} "
            + "=" * 25
        )

        # ======================================================
        # 1. GENERATE / REFINE
        # ======================================================

        if i == 0:

            print("\n>>> STAGE 1: GENERATING initial code...")

            response = llm.invoke(
                message_history
            )

        else:

            print(
                "\n>>> STAGE 1: REFINING code "
                "based on previous critique..."
            )

            message_history.append(
                HumanMessage(
                    content=(
                        "Improve the previous code using the "
                        "critique provided. "
                        "Return only the updated Python code."
                    )
                )
            )

            response = llm.invoke(
                message_history
            )

        current_code = response.content

        print(
            f"\n--- Generated Code (v{i + 1}) ---\n"
        )

        print(current_code)

        message_history.append(response)


        # ======================================================
        # 2. REFLECT
        # ======================================================

        print(
            "\n>>> STAGE 2: REFLECTING on the generated code..."
        )

        reflector_prompt = [
            SystemMessage(
                content="""
You are a senior software engineer and an expert in Python.

Your role is to perform a meticulous code review.

Critically evaluate the provided Python code based on the
original task requirements.

Check for:

- bugs
- incorrect logic
- missing edge cases
- missing validation
- poor Python style
- unclear documentation
- unnecessary complexity

If the code is perfect and meets all requirements, respond with:

CODE_IS_PERFECT

Otherwise, provide a concise bulleted list of critiques.
"""
            ),
            HumanMessage(
                content=f"""
Original Task:

{task_prompt}

Code to Review:

{current_code}
"""
            ),
        ]

        critique_response = llm.invoke(
            reflector_prompt
        )

        critique = critique_response.content


        # ======================================================
        # 3. STOPPING CONDITION
        # ======================================================

        if "CODE_IS_PERFECT" in critique:

            print(
                "\n--- Critique ---"
            )

            print(
                "No further critiques found. "
                "The code is satisfactory."
            )

            break


        print(
            "\n--- Critique ---\n"
        )

        print(critique)


        # ======================================================
        # ADD CRITIQUE FOR NEXT ITERATION
        # ======================================================

        message_history.append(
            HumanMessage(
                content=f"""
Critique of the previous code:

{critique}

Use this critique to improve the code in the next iteration.
"""
            )
        )


    # ==========================================================
    # FINAL RESULT
    # ==========================================================

    print(
        "\n"
        + "=" * 30
        + " FINAL RESULT "
        + "=" * 30
    )

    print(
        "\nFinal refined code after the reflection process:\n"
    )

    print(current_code)


# ==============================================================
# RUN
# ==============================================================

if __name__ == "__main__":
    run_reflection_loop()
