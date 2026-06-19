from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from config import GROQ_API_KEY, LLAMA_MODEL_ID, QWEN_MODEL_ID, GPT_MODEL_ID


class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    category: str = Field(description="Category of the inquiry (e.g., billing, technical, general)")
    action: str = Field(description="Recommended action for the support rep")
    response: str = Field(description="Suggested response to the user")


json_parser = JsonOutputParser(pydantic_object=AIResponse)

def initialize_model(model_id):
    return ChatGroq(
        model=model_id,
        api_key=GROQ_API_KEY,
        temperature=0
    )


llama_llm = initialize_model(LLAMA_MODEL_ID)
qwen_llm = initialize_model(QWEN_MODEL_ID)
gpt_llm = initialize_model(GPT_MODEL_ID)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}\n{format_prompt}"),
    ("user", "{user_prompt}")
])

def get_ai_response(model, system_prompt, user_prompt):
    chain = prompt_template | model | json_parser
    return chain.invoke({
        'system_prompt': system_prompt, 
        'user_prompt': user_prompt, 
        'format_prompt': json_parser.get_format_instructions()
    })


def llama_response(system_prompt, user_prompt):
    return get_ai_response(llama_llm, system_prompt, user_prompt)

def qwen_response(system_prompt, user_prompt):
    return get_ai_response(qwen_llm, system_prompt, user_prompt)

def gpt_response(system_prompt, user_prompt):
    return get_ai_response(gpt_llm, system_prompt, user_prompt)
