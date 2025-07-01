import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class TableRepresentations(BaseModel):
    summary: str = Field(
        description="A concise, professional summary of the table's content, purpose, and key data points.")
    hypothetical_questions: list[str] = Field(
        description="A list of 3-5 hypothetical, specific questions a financial analyst might ask about this table."
    )


def get_table_representation_chain():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in your .env file.")

    llm = ChatOpenAI(
        # based on Artificial Analysis, I would recommend llama 4 maverick for the actual system
        # much cheaper, very low TTFT and intelligent enough
        # using mistral for this because openrouter does not support JSON output and maverick is unreliable without it
        model="mistralai/mistral-small-3.2-24b-instruct-2506:free",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=1024
    )

    prompt_text = """You are a world-class financial analyst AI. Your task is to analyze the provided financial table and generate a structured representation of it.

    Based on the table content below, provide a concise summary and a list of 3-5 specific, hypothetical questions that this table can answer.

    Return ONLY a valid JSON object with a "summary" and a "hypothetical_questions" key.

    YOU ARE ABSOLUTELY NOT TO INCLUDE ANYTHING LIKE A ``` OR ANY OTHER TOKENS BESIDES THOSE OF THE JSON.

    ENSURE ALL HYPOTHETICAL QUESTIONS ARE PRESENT. 

    Table Content:
    {table_content}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    parser = JsonOutputParser(pydantic_object=TableRepresentations)

    chain = prompt | llm | parser

    return chain

def get_MRI(table_content) -> TableRepresentations:
    try:
        generation_chain = get_table_representation_chain()

        result = generation_chain.invoke({"table_content": table_content})

    except Exception as e:
        print("\n--- ERROR ---")
        print(f"An error occurred during chain invocation: {e}")
        result = None

    return result