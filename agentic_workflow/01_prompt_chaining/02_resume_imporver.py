"""
PROBLEM STATEMENT
-----------------
Build a multi-step resume processing pipeline where each step is a
separate LLM call and the output of one step is passed to the next.

Current pipeline:

Step 1 - Candidate Extraction
    Input:
        Raw resume text

    Task:
        - Extract relevant professional/technical skills
        - Calculate total years of professional experience

    Output:
        CandidateProfile Pydantic object
        {
            skills: list[str],
            experience_years: float
        }


Step 2 - Resume Bullet Rewriting
    Input:
        - Original resume text
        - Skills extracted in Step 1
        - Experience years extracted in Step 1

    Task:
        Generate 3-4 stronger resume experience bullets using the
        candidate's actual experience and skills.

    Output:
        Rewritten resume bullets as text.


APPROACH
--------
The pipeline uses LangChain Runnable composition:

Raw Resume
    |
    v
Candidate Extraction Prompt
    |
    v
LLM Call #1
    |
    v
PydanticOutputParser
    |
    v
CandidateProfile
    |
    | Keep original resume using RunnablePassthrough
    v
Prepare Step 2 Input
    |
    v
Resume Rewrite Prompt
    |
    v
LLM Call #2
    |
    v
StrOutputParser
    |
    v
Rewritten Resume Bullets


RunnablePassthrough.assign():
    Preserves the original resume input while adding the output
    produced by the candidate extraction chain.

RunnableLambda():
    Reshapes the combined data into the exact variables required
    by the second prompt.
"""


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser,
)

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)


load_dotenv()


model = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0.3,
)


class CandidateProfile(BaseModel):

    skills: list[str] = Field(
        description="Relevant professional and technical skills of the candidate."
    )

    experience_years: float = Field(
        description=(
            "Calculate total years of professional work experience "
            "from employment dates. Avoid double-counting overlapping roles. "
            "Exclude education and non-professional projects. "
            "Return the result rounded to one decimal place."
        )
    )


candidate_parser = PydanticOutputParser(
    pydantic_object=CandidateProfile
)


candidate_extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a resume analyzer.

Your task is to:
1. Extract relevant professional and technical skills.
2. Calculate total years of professional work experience.

{format_instructions}
"""
    ),
    (
        "user",
        """
Resume:

{resume_text}
"""
    )
])


candidate_extraction_prompt = candidate_extraction_prompt.partial(
    format_instructions=candidate_parser.get_format_instructions()
)


candidate_extraction_chain = (
    candidate_extraction_prompt
    | model
    | candidate_parser
)


resume_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a professional resume rewriter.

Analyze the original resume and use the extracted candidate information.

Original Resume:
{resume_text}

Extracted Skills:
{skills}

Professional Experience:
{experience_years} years

Create 3-4 strong interview-impact resume bullets.

Do not invent metrics, technologies, achievements, or experience
that are not supported by the original resume.
"""
    )
])


resume_rewrite_chain = (
    resume_rewrite_prompt
    | model
    | StrOutputParser()
)


def prepare_rewrite_input(data):

    return {
        "resume_text": data["resume_text"],
        "skills": data["candidate_profile"].skills,
        "experience_years": data["candidate_profile"].experience_years,
    }


resume_pipeline = (
    RunnablePassthrough.assign(
        candidate_profile=candidate_extraction_chain
    )
    | RunnableLambda(prepare_rewrite_input)
    | resume_rewrite_chain
)


resume_text = """
Naman Bhalla

PROFESSIONAL SUMMARY
Data and AI professional with experience building machine learning,
generative AI, and retrieval-augmented generation applications.

SKILLS
- Python
- SQL
- Machine Learning
- Generative AI
- LangChain
- RAG
- ChromaDB
- Vector Search
- BM25
- OpenAI APIs
- Pandas
- Scikit-learn

EXPERIENCE

AI Engineer
ABC Technologies
2023 - Present

- Developed retrieval-augmented generation applications using LangChain.
- Built hybrid search pipelines combining BM25 and vector search.
- Implemented vector databases using ChromaDB.
- Integrated OpenAI models for embeddings and text generation.

Data Analyst
XYZ Solutions
2021 - 2023

- Analyzed business datasets using Python, Pandas, and SQL.
- Built machine learning models using Scikit-learn.

EDUCATION
Bachelor of Technology in Computer Science
XYZ University
2021
"""


result = resume_pipeline.invoke({
    "resume_text": resume_text
})


print(result)
