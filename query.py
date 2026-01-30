import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm=ChatOpenAI(
   model="meta-llama/llama-3.1-8b-instruct",    
   base_url="https://openrouter.ai/api/v1",
   temperature=0.9,
)

#Loading embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

#loading faiss index
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
while True:
    query=input("Enter your query: ")
    if(query.lower() in ['exit','quit']):
        break
    else:
        docs = vectorstore.similarity_search(query, k=8)
        context=""
        for d in docs:
            context += f"\n\nSource: {d.metadata}\n{d.page_content}"
        # print(context)
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer ONLY using the provided {context}. If the answer is not in the {context}, say 'Not found in uploaded documents'."),
                ("human", "Question: {query}")])
        prompt=template.format(context=context,query=query)
        result=llm.invoke(prompt)
        print(result.content)


