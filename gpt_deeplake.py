#!python3 -m pip install --upgrade langchain deeplake openai

import os
from getpass import getpass
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


from langchain.vectorstores import DeepLake


os.environ['OPENAI_API_KEY'] = getpass()

os.environ['ACTIVELOOP_TOKEN'] = getpass.getpass('Activeloop Token:')


root_dir = '../../../..'

docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith('.py') and '/.venv/' not in dirpath:
            try: 
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e: 
                pass
print(f'{len(docs)}')



text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
print(f"{len(texts)}")






embeddings = OpenAIEmbeddings()
db = DeepLake.from_documents(texts, embeddings, dataset_path=f"hub://{DEEPLAKE_ACCOUNT_NAME}/langchain-code")


db = DeepLake(dataset_path=f"hub://{DEEPLAKE_ACCOUNT_NAME}/langchain-code", read_only=True, embedding_function=embeddings)


retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 20
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 20


def filter(x):
    # filter based on source code
    if 'something' in x['text'].data()['value']:
        return False
    
    # filter based on path e.g. extension
    metadata =  x['metadata'].data()['value']
    return 'only_this' in metadata['source'] or 'also_that' in metadata['source']

### turn on below for custom filtering
# retriever.search_kwargs['filter'] = filter

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

model = ChatOpenAI(model='gpt-3.5-turbo') # 'ada' 'gpt-3.5-turbo' 'gpt-4',
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)



questions = [
    "What is the class hierarchy?",
    # "What classes are derived from the Chain class?",
    # "What classes and functions in the ./langchain/utilities/ forlder are not covered by unit tests?",
    # "What one improvement do you propose in code in relation to the class herarchy for the Chain class?",
] 
chat_history = []

for question in questions:  
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")