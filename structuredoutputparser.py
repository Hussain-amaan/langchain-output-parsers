from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()


model = model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

schema=[
    ResponseSchema(name='fact_1', description="fact 1 of the topic  "),
    ResponseSchema(name='fact_2', description="fact 2 of the topic  "),
    ResponseSchema(name='fact_3', description="fact 3 of the topic  ")
]

parser=StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='give 3 facts about the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}

)


chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)