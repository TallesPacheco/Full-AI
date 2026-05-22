from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://www.google.com/")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

chunks = splitter.split_documents(docs)

for chunk in chunks:
    print(chunk)
    print("-" * 30)