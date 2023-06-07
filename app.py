import os
from langchain.llms import OpenAI
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

project_path = "Your Project PATH"
os.environ['OPENAI_API_KEY'] = 'Your API Key'
def app(index):
    llm = OpenAI(verbose=True)
    st.title('🦜🔗 PDF Buddy')
    prompt = st.text_input('Input your prompt here')

    if prompt:
        response = index.query(llm=llm, question=prompt, chain_type="stuff")
        st.write(response)

def save_uploadedfile(uploadedfile, project_path):
     with open(os.path.join(f"{project_path}/docs",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("File Loaded!")

def pdfloader(pdf, project_path):
    if os.path.exists(f"{project_path}/docs/{pdf.name}") == False:
        save_uploadedfile(pdf, project_path)
        print("file saved and loaded")
    if os.path.exists(f"{project_path}/docs/{pdf.name}") == True:
        st.success("File Loaded!")
        print("file already exists, loading it up")
        
    loader = PyPDFLoader(f"{project_path}/docs/{pdf.name}")
    index = VectorstoreIndexCreator(
        text_splitter = CharacterTextSplitter(chunk_size = 2000, chunk_overlap = 0),
        embedding = OpenAIEmbeddings(),
        vectorstore_cls = Chroma
    ).from_loaders([loader])
    return index

pdf = st.file_uploader("Upload a PDF file/book", type="pdf")
if pdf:
    index = pdfloader(pdf, project_path)
    app(index)