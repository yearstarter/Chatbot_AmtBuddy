from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain

import streamlit as st

# Import document loaders
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import TextLoader

# Import OpenAiEmbeddings and Chroma for vectorstores and embeddings 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# import modules for Question Answering
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# import modules for chat memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import os
# from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

from apikeys import apikey_OpenAI

##############################################################################

def main():
    os.environ['OPENAI_API_KEY'] = apikey_OpenAI

    st.set_page_config(page_title="AmtBuddy", page_icon="💬")

    ################
    # Main window
    st.header("Getting started in Germany with AmtBuddy :de:")
    st.text("""
    This chatbot will help you with common immigration topics, 
    like work permission, finance, renting, insurance...
    The answer to your question will be provided in English""")
    user_question = st.text_input("Ask your question here:")
    
    if user_question:
             
        #####
        # searching for the relevant chunks in vector store
        persist_directory = 'Documents/...' # path to local vector store 
        embedding = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        
        retriever_similarity = vectordb.as_retriever(
            search_type = "similarity", 
            search_kwargs={"k": 5}) 
        
        #####
        # generating an answer using the relevant chunks and user_input
        chat_llm_name = "gpt-3.5-turbo"
        chat_llm = ChatOpenAI(
            model_name=chat_llm_name, 
            temperature=0  # low variability gives a factual answer 
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        rel_docs = vectordb.similarity_search(user_question, k=3)
        
        system_template = """Use only the following pieces of context to answer the user's question. 
        If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
        ----------------
        {context}"""

        # Create the chat prompt template
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}?")
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)
        
        
        qa_chain_memory = ConversationalRetrievalChain.from_llm(
            chat_llm,
            retriever=retriever_similarity,
            # chain_type="map_rerank",
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            # verbose=True
        )
        st.write(qa_chain_memory.run(user_question))

        with st.expander('See some relevant text chunks from trusted documents in German'): 
            for d in rel_docs:
                st.write(d.page_content)
        
    ################    
    # Sidebar

    with st.sidebar:

        st.subheader("Upload a document or letter you want to chat with")

        pdf = st.file_uploader(
            "Please upload only PDFs",
            type="pdf"
        )

        with st.spinner(text='Progress'):

            if pdf is not None:
                pdf_reader = PdfReader(pdf)
                text = "" 
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", "(?<=\. )", " ", ""],
                    length_function=len
                )

                user_chunks = text_splitter.split_text(text)
                user_embeddings = OpenAIEmbeddings()
                db_user = FAISS.from_texts(user_chunks, user_embeddings)

                user_input_sidebar = st.text_input("Ask a question here:")
                if user_input_sidebar:
                    user_docs  = db_user.similarity_search(user_input_sidebar)
                    llm = ChatOpenAI(temperature=0.1)  # also works with OpenAI(), but seems to provide less words
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=user_docs, question=user_input_sidebar)
                    
                    st.write(response)

                st.button("Process")

# Dieser Code testet, ob die application direkt ausgeführt und nicht erst importiert wird. 
# Wenn das erfüllt ist, dann wird erst die main() Funktion ausgeführt
if __name__ == '__main__':
    main()
