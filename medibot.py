import streamlit as st
import os
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- Load Environment Variables ---
# This line loads your GOOGLE_API_KEY from the .env file
load_dotenv()

# --- Constants ---
DB_FAISS_PATH = "vectorstore/db_faiss"
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# --- Caching Functions for Performance ---

@st.cache_resource
def load_llm():
    """
    Loads and caches the Gemini 1.5 Flash model.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    return llm

@st.cache_resource
def load_vector_store():
    """
    Loads and caches the FAISS vector store and the embedding model.
    """
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def create_qa_chain(_db, _llm):
    """
    Creates and returns the RetrievalQA chain.
    """
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# --- Main Streamlit App ---

def main():
    """
    The main function that runs the Streamlit application.
    """
    st.set_page_config(page_title="MediQ RAG", page_icon="⚕️")
    st.title("MediQ RAG ⚕️")
    st.subheader("Your Personal Medical Assistant")

    st.warning(
        "**Disclaimer:** I am an AI assistant, not a medical professional. "
        "The information I provide is for educational purposes only and is not a substitute for professional medical advice. "
        "Always consult a qualified healthcare provider for any health concerns."
    )
    st.caption("My knowledge is based exclusively on 'The Gale Encyclopedia of Medicine, Second Edition'.")
   

    # Load the necessary components
    db = load_vector_store()
    llm = load_llm()
    qa_chain = create_qa_chain(db, llm)

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_query := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_query)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Show a thinking spinner while processing
            with st.spinner("Thinking..."):
                # Get the response from the QA chain
                response = qa_chain.invoke({'query': user_query})
                full_response = response['result']
                
                # Display the main answer
                st.markdown(full_response)
                
                # Optionally, display the source documents in an expander
                with st.expander("View Sources"):
                    for doc in response["source_documents"]:
                        st.write(f"**Page {doc.metadata.get('page', 'N/A')}:**")
                        st.info(f"{doc.page_content[:250]}...")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()

