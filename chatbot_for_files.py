# chatbot using Langchain, OpenAI (API key required),
# and existing Pinecone index (API key required).

# Import required libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredWordDocumentLoader, PyMuPDFLoader, UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone, Chroma
from langchain.chains import ConversationalRetrievalChain
from colorama import init, Fore, Style
import os
import pinecone

# Set up OpenAI API key (from .bashrc, Windows environment variables, .env)
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Set up OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Use Open AI LLM with gpt-3.5-turbo.
# Set the temperature to be 0 if you do not want it to make up things
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True,
                 openai_api_key=OPENAI_API_KEY)


# Set up Pinecone env
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_API_ENV = os.environ['PINECONE_API_ENV']
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Get user input of whether to use Pinecone or not
r = input(Fore.GREEN + 'Do you want to use Pinecone index? (y/n): ' + Style.RESET_ALL)
if r.lower() == 'y' and PINECONE_API_KEY != '':
    pinecone_index_name = input(
        Fore.GREEN + "Enter your Pinecone index: " + Style.RESET_ALL)
    use_pinecone = True
else:
    chroma_collection_name = input(
        Fore.GREEN +
        'Not using Pinecone or empty Pinecone API key provided. Using Chroma. Enter Chroma collection name: ' + Style.RESET_ALL)
    use_pinecone = False
    persist_directory = "./vectorstore"

# Get user input of whether to ingest files or using existing vector store
r = input(Fore.GREEN +
          'Do you want to ingest the file(s) in ./docs/? (y/n): ' + Style.RESET_ALL)
if r.lower() == 'y':

    file_path = './docs/'
    all_texts = []
    n_files = 0
    n_char = 0
    n_texts = 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50)
    for filename in os.listdir(file_path):
        file = os.path.join(file_path, filename)
        if os.path.isfile(file):
            if file.endswith('.docx'):
                loader = UnstructuredWordDocumentLoader(file)
            elif file.endswith('.txt'):
                loader = UnstructuredFileLoader(file)
            elif file.endswith('.pdf'):
                loader = PyMuPDFLoader(file)
            else:
                continue
            data = loader.load()
            texts = text_splitter.split_documents(data)
            n_files += 1
            n_char += len(data[0].page_content)
            n_texts += len(texts)
            all_texts.extend(texts)
    print(
        f'Loaded {n_files} file(s) with {n_char} characters, and split into {n_texts} split-documents.')

    if use_pinecone:
        docsearch = Pinecone.from_texts(
            [t.page_content for t in all_texts], embeddings, index_name=pinecone_index_name)  # add namespace=pinecone_namespace if provided
    else:
        docsearch = Chroma.from_documents(
            all_texts, embeddings, collection_name=chroma_collection_name, persist_directory="./vectorstore")
else:
    if use_pinecone:
        # Load the pre-created Pinecone index.
        # The index which has already be stored in pinecone.io as long-term memory

        if pinecone_index_name in pinecone.list_indexes():
            docsearch = Pinecone.from_existing_index(
                pinecone_index_name, embeddings)  # add namespace=pinecone_namespace if provided
            index_client = pinecone.Index(pinecone_index_name)
            # Get the index information
            index_info = index_client.describe_index_stats()
            namespace_name = ''
            n_texts = index_info['namespaces'][namespace_name]['vector_count']
        else:
            raise ValueError('''Cannot find the specified Pinecone index.
							Create one in pinecone.io or using 
							pinecone.create_index(
								name=index_name, dimension=1536, metric="cosine", shards=1)''')
    else:
        docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings,
                           collection_name=chroma_collection_name)
        n_texts = docsearch._client._count(
            collection_name=chroma_collection_name)

# print(len(docsearch))
# Ask if the source document should be printed out.
# If yes, the ouptput could be very long
print_source = input(Fore.GREEN +
                     "You want to also print texts from two most relevant source documents? y/n: "
                     + Style.RESET_ALL)

# number of sources (split-documents when ingesting files); default is 4
k = min([20, n_texts])

# Set up retriever
retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": k}, include_metadata=True)

# Set up ConversationalRetrievalChain
CRqa = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, return_source_documents=True)


def chatbot_loop():
    # Get a response from the retriever based on the user input and chat history
    def get_response(query, chat_history):
        result = CRqa({"question": query, "chat_history": chat_history})
        return result['answer'], result['source_documents']

    # Initialize chat history
    chat_history = []
    # Initialize color
    init()
    # Start the chat loop
    while True:
        # Get user input
        query = input(
            Fore.GREEN + "Enter your question; enter 'exit' to exit: " + Style.RESET_ALL)
        if query.lower() == 'exit':
            break
        # Generate a reply based on the user input and chat history
        reply, source = get_response(query, chat_history)
        print(reply)
        # Print the two most relevant sources
        # (only the first 400 characters for brevity here) if needed
        if print_source.lower() == 'y':
            for i, source_i in enumerate(source):
                if i < 2:
                    if len(source_i.page_content) > 400:
                        page_content = source_i.page_content[:400]
                    else:
                        page_content = source_i.page_content
                    if source_i.metadata:
                        metadata_source = source_i.metadata['source']
                        print(Fore.GREEN + "Source document info and content: " +
                              Style.RESET_ALL, metadata_source, ": ", page_content)
                        print(source_i.metadata)
                    else:
                        print(Fore.GREEN + "Source document content: " +
                              Style.RESET_ALL, page_content)

        # Update the chat history with the user input and system response
        chat_history.append((query, reply))


chatbot_loop()
