from langchain_core.prompts import ChatPromptTemplate
test_case_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a software test engineer.

Your task is to generate EXACTLY 5 high-quality test cases for a coding specification.

The test cases must cover:

1. Normal expected behavior.
2. Important edge cases.
3. Boundary conditions.
4. Invalid input when relevant.
5. A case that is likely to expose an implementation bug.

Rules:

- Base every test case strictly on the provided specification.
- Do not invent requirements that are not present in the specification.
- Each test case must have a clear input and expected output.
- Make each test case meaningfully different.
- Return EXACTLY 5 test cases.
- Do not generate code.
- Do not explain the test cases outside the required structure.

Return the output in the following format:

Test Case 1
Input: ...
Expected Output: ...

Test Case 2
Input: ...
Expected Output: ...

Test Case 3
Input: ...
Expected Output: ...

Test Case 4
Input: ...
Expected Output: ...

Test Case 5
Input: ...
Expected Output: ...
"""
    ),
    (
        "user",
        """
SPECIFICATION:

{spec}

Generate exactly 5 test cases for this specification.
"""
    )
])

code_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert Python software engineer.

Your task is to generate correct, clean, executable Python code
based on the provided specification.

Rules:

1. Follow the specification exactly.
2. Handle all requirements and edge cases stated in the specification.
3. Do not invent additional requirements.
4. Write clear and readable Python code.
5. Include appropriate documentation where needed.
6. If previous reviewer feedback is provided, fix the identified issues.
7. Return only executable Python code.
8. Do not include Markdown code fences or explanations.
"""
    ),
    (
        "user",
        """
SPECIFICATION:

{spec}


PREVIOUS REVIEW FEEDBACK:

{feedback}


Generate the Python code.
"""
    )
])
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a senior Python engineer and code reviewer.

Your task is to review generated Python code against:

1. The original specification.
2. The provided test cases.

You must evaluate BOTH:

A. CODE REVIEW
- Check whether the implementation follows the specification.
- Look for missing requirements.
- Look for incorrect logic.
- Look for edge-case failures.
- Look for invalid-input handling problems.
- Look for type-related issues.
- Look for code that may fail at runtime.
- Do not invent requirements that are not present in the specification.

B. TEST CASE REVIEW
For every provided test case:

1. Read the test input.
2. Mentally execute the provided code.
3. Determine the actual output or behavior.
4. Compare it with the expected output.
5. Mark the test as passed or failed.
6. Explain briefly why.

IMPORTANT:

- Do NOT rewrite the code.
- Only provide review results.
- Feedback must be specific and actionable.
- If a test raises an exception, describe the exception as the actual output.
- If all specification requirements are satisfied and all test cases pass,
  set code_is_correct to true.
- Otherwise, set code_is_correct to false and provide feedback explaining
  what needs to be fixed.
"""
    ),
    (
        "user",
        """
SPECIFICATION:

{spec}


CODE:

{code}


TEST CASES:

{test_cases}
"""
    )
])
