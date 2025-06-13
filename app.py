## RAG Q & A conversation app with pdf upload and chat history

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory 
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = st.secrets["HF_TOKEN"]
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", api_key=HF_TOKEN)

## Setup the streamlit app
st.set_page_config(page_title="PDF Chatbot")
st.title("PDF Chatbot with chat history")
#st.write("Upload PDF and chat about the content")

## Input the GROQ API Key
api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

## Check if the API is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="gemma2-9b-it")

    session_id = st.text_input("Session ID", value="default")

    ## Manage a chat history statefully

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

    ## Process the uploaded files

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

    ## Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents = splits, embedding = embeddings, persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = ("""
            Given a chat history and the latest user question which might reference context in the chat history, Formulate a standalone
            question which can be understood without the chat history, Do not answer the question, just reformulate it if needed otherwise return it as is.
            """)
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriver = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        ## Answer the question prompt
        system_prompt = ("""
                    you are an assistant for question answering task. use the following retrieved context to answer the question.
                    If you dont know the answer, say that you dont know. Use three sentences maximum and keep the answer concise.
                    \n\n
                    {context}
                """)
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        q_a_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriver, q_a_chain)

        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        user_input = st.text_input("Ask a question about the content of the PDF")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversation_rag_chain.invoke({"input": user_input},
                                                     config={
                                                         "configurable": {"session_id": session_id}
                                                         },
                                                        )
            #st.write(st.session_state.store)
            st.write("Assistant:", response["answer"])
            with st.expander("View Chat History"):
                st.write("Chat history:", session_history.messages)
                    
else:
    st.warning("Please provide your GROQ API Key to use the app.")
    st.write("You can get your API key from [GROQ](https://groq.com/).")
