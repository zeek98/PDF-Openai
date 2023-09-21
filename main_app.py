from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from sentence_transformers import SentenceTransformer
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.chains import ConversationChain
import pinecone
import openai
import os

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENVIRONMENT')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index('dp-index')

# Initialize an empty cache for PDFs
pdf_cache = {}

# Function to retrieve text data from the cache or load it from a PDF if not present
def get_text_from_cache(pdf_identifier, pdf_path):
    if pdf_identifier in pdf_cache:
        return pdf_cache[pdf_identifier]
    else:
        # Load and extract text data from the PDF if not in cache
        text_data = load_and_extract_pdf_text(pdf_path)
        # Store the text data in the cache for future use
        pdf_cache[pdf_identifier] = text_data
        return text_data

# Function to simulate loading and extracting text data from a PDF
def load_and_extract_pdf_text(pdf_path):
    # Replace this with your logic to load and extract text data from a PDF
    # You can use libraries like PyPDF2 or pdfplumber for text extraction
    # For simplicity, we'll use a dummy function here
    return f"Text data extracted from PDF: {pdf_path}"

# Utility functions
def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

# Function to display messages
def message(text, is_user=False, key=None):
    if is_user:
        st.text(f"User: {text}")
    else:
        st.text(f"Bot: {text}")

# Main application
def main():
    st.set_page_config(page_title="Chat and PDF QA")
    st.header("Chat and PDF QA 💬📚")

    # Option to choose between Chat and PDF
    option = st.radio("Choose an option:", ("Chat", "PDF"))

    if option == "Chat":
        # Chat functionality
        st.subheader("Chat")
        if 'responses' not in st.session_state:
            st.session_state['responses'] = ["How can I assist you?"]

        if 'requests' not in st.session_state:
            st.session_state['requests'] = []

        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

        if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

        system_msg_template = SystemMessagePromptTemplate.from_template(
            template="""Answer the question as truthfully as possible using the provided context, 
            and if the answer is not contained within the text below, say 'I don't know'""")

        human_msg_template = HumanMessagePromptTemplate.from_template("{input}")

        prompt_template = ChatPromptTemplate.from_messages(
            [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

        conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm,
                                         verbose=True)

        # Container for chat history
        response_container = st.container()
        # Container for text box
        textcontainer = st.container()

        with textcontainer:
            query = st.text_input("Query: ", key="input")
            if query:
                with st.spinner("typing..."):
                    conversation_string = get_conversation_string()
                    refined_query = query_refiner(conversation_string, query)
                    st.subheader("Refined Query:")
                    st.write(refined_query)
                    context = find_match(refined_query)
                    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)

        with response_container:
            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])):
                    message(st.session_state['responses'][i], key=str(i))
                    if i < len(st.session_state['requests']):
                        message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

    elif option == "PDF":
        # PDF functionality
        st.subheader("PDF")
        pdf = st.file_uploader("Upload your PDF", type="pdf")

        if pdf is not None:
            pdf_identifier = "pdf_cache_" + str(hash(pdf))
            text_data = get_text_from_cache(pdf_identifier, pdf)
            
            user_question = st.text_input("Ask a question about your PDF:")
            if user_question:
                # Use the extracted text data (text_data) for question answering
                # You can use your existing code to perform QA using the text_data
                # and display the response
                st.write("Perform QA using the text data extracted from the PDF.")
                
if __name__ == '__main__':
    main()
