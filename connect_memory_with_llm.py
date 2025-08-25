
import os
from dotenv import load_dotenv, find_dotenv


from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# This line loads your GOOGLE_API_KEY from the .env file
load_dotenv(find_dotenv())

# Step 1: Setup LLM (Google Gemini)
def load_llm():
    """
    Loads the Gemini 1.5 Flash model.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                               temperature=0.5)
    return llm

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
def set_custom_prompt(custom_prompt_template):
    """
    Sets up the custom prompt template for the QA chain.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load the FAISS vector database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create the QA chain with the new Gemini LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),  # Use the new Gemini LLM loader
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Start the interactive query session
if __name__ == "__main__":
    while True:
        user_query = input("Write Query Here (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        
        # Invoke the chain with the user's query
        response = qa_chain.invoke({'query': user_query})
        
        # Print the results
        print("\nRESULT:")
        print(response["result"])
        print("\nSOURCE DOCUMENTS:")
        # Loop through source documents for cleaner printing
        for doc in response["source_documents"]:
            print(f"- Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:150]}...")
        print("-" * 50)

