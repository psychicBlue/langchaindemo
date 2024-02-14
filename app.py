from customllm import CustomLLM
from customllm import default_llm
from custom_embedders import CustomChromaEmbedder
from custom_embedders import default_embedder
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

#question = "Write me a SQL query to get the average age for all active users using the following context"

#question = "Write a SQL query to retrieve total number of customers."
question = "Which table consists the list of tracks?"
dialect = "SQL Server"
top_k = 1

template = """You are an agent designed to generate SQL queries.
Given an input question, create a syntactically correct {dialect} query to run.

This is the following input question: {question}

Only generate a SQL query based on the table schemas present here: {context}

If the question does not seem related to the database, just return "I don't know" as the answer.
Avoid generating additional questions or answers.
"""

# embedder = CustomChromaEmbedder()
embedder = default_embedder
db = Chroma(persist_directory="./chromadb", embedding_function=embedder)
retriever = db.as_retriever(search_kwargs={"k": top_k})

prompt = PromptTemplate(template=template, input_variables=['context', 'dialect', 'top_k', 'question'])

# llm = CustomLLM()
llm = default_llm
# chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
chain = (
    {"context": retriever, "top_k": RunnablePassthrough(), "dialect":RunnablePassthrough(), "question": RunnablePassthrough()}
    #{"question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(chain.invoke(question))

