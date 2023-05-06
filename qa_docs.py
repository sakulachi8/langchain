from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import magic
import os
import nltk

# os.environ['OPENAI_API_KEY'] = '...'

# nltk.download('averaged_perceptron_tagger')

# pip install unstructured
# Other dependencies to install https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/unstructured_file.html
# pip install python-magic-bin
# pip install chromadb


loader = DirectoryLoader('../data/', glob='**/*.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
docsearch = Chroma.from_documents(texts, embeddings)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)
query = "What did McCarthy discover?"
qa.run(query)


# Source File too
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
query = "What did McCarthy discover?"
result = qa({"query": query})
result['result']

