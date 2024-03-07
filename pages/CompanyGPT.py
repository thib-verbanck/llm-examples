import streamlit as st 
import os
import anthropic  
from langchain.agents import initialize_agent, AgentType  
from langchain.callbacks import StreamlitCallbackHandler 
from langchain.chat_models import ChatOpenAI 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake #docs: https://api.python.langchain.com/en/v0.0.343/vectorstores/langchain.vectorstores.deeplake.DeepLake.html
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from docx import Document
import PyPDF2
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.agents import create_react_agent, Tool
from langchain.agents import initialize_agent, AgentType


#------------------------------------------------------
# INITALIZE FUNCTIONS   
#------------------------------------------------------

def llm():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)
    return llm

def db():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    my_activeloop_org_id = "thibverbanck"
    my_activeloop_dataset_name = "RAG-test-3"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
    return db
    
def retriever(llm, db):
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
        )
    return retrieval_qa

def tools(retriever):
    tools = [
    Tool(
        name="Retrieval QA System",
        func=retriever.run,
        description="Useful for answering questions about Thibaut Verbanck and Ella Vieren."
        ),
    ]
    return tools
    
def agent(tools, llm, verbose_true):
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    agent = initialize_agent(
        tools,
        llm,
        agent=agent_type,
        verbose=verbose_true
    )
    return agent

my_llm = llm()
my_db = db()
my_retriever = retriever(my_llm, my_db)
my_tools = tools(my_retriever)

#------------------------------------------------------
# LAY OUTING
#------------------------------------------------------

# Creating a sidebar in the Streamlit app
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="file_qa_api_key", type="password")

# Setting the title of the Streamlit app
st.title("Chat with your files")

#------------------------------------------------------
# READ DOCUMENTS FROM DIRECTORY PATH
#------------------------------------------------------

# Ask for directory path
directory_path = st.text_input("Enter the directory path", type="text")

# Define function to read word file 
def read_word_file(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Define function to read pdf file
def read_pdf_file(file_path):
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    full_text = []
    for page_num in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page_num]
        full_text.append(page_obj.extract_text())
    pdf_file_obj.close()
    return '\n'.join(full_text)

# Check if the directory path exists
if os.path.exists(directory_path):
    directory = 'G:/Mijn Drive/RAG-test-1/RAG-test-1/Files'

    # Define a string to check all output of files
    knowledge_base = ""

    # Read files to text, save output in knowledge_base string
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith('.docx'):
            text = read_word_file(file_path)
            knowledge_base += text + " "
        elif filename.endswith('.pdf'):
            text = read_pdf_file(file_path)
            knowledge_base += text + " "
        else:
            print(f"Unsupported file type: {filename}")
    
    knowledge_base = knowledge_base.strip()
    print(knowledge_base)

else:
    st.error("The provided directory path does not exist.")

if len(knowledge_base) > 0: 
    knowledge_base_available = True
else: knowledge_base_available = False

#------------------------------------------------------
# SAVE DOCUMENTS FROM DIRECTORY PATH
#------------------------------------------------------

# Define text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Define chuncks
chunks = text_splitter._split_text(knowledge_base, separators= ' ')  ##in de toekomst moeten we hier document splitter gaan gebruiken, daarna zelfs semantische split toevoegen

# Print chuncks for output check
for chunk in chunks:
    print(chunk)
    print("----------------------------")

# Define keys
os.environ['ACTIVELOOP_TOKEN'] = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpZCI6InRoaWJ2ZXJiYW5jayIsImFwaV9rZXkiOiJ0RnVVZzR6S2lya05TTU5fMVRubndWWnd3Zmp4eGIzMUtlQjg5dVZlUDIzM3kifQ."

# Define embeddings model
my_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key = openai_api_key)

# Create Deep Lake dataset
my_activeloop_org_id = "thibverbanck" 
my_activeloop_dataset_name = "RAG-test-4"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=my_embeddings)

# Add documents to our Deep Lake dataset
db.add_texts(chunks)

#------------------------------------------------------
# CHATBOT
#------------------------------------------------------

# Adding a text input field for user questions
question = st.text_input(
    "Ask me something",
    placeholder="Your question goes here.",
    disabled=not knowledge_base_available,
)

# Checking if a file is uploaded, a question is asked, and an OpenAI API key is provided
if knowledge_base_available and question and not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")

# Checking if a file is uploaded, a question is asked, and an OpenAI API key is provided
if knowledge_base_available and question and openai_api_key:
    response = agent(my_tools, my_llm, False).invoke(question)
    st.write("### Answer")
    st.write(response)