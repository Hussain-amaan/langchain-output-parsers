from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


model = model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


class person(BaseModel):
    name: str=Field(description="Name of person")
    age: int=Field(description="Enter age of the person")
    city: str=Field(description="Entr the city of person")


parser=PydanticOutputParser()

template=PromptTemplate(
    template='generate the name , age and city of a fictional {place} person \n ' \
    '{format_instruction} ',
    input_variables=['palce'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)  

prompt=template.invoke({'place':'indian'})
result=model.invoke(prompt)
print(result)
