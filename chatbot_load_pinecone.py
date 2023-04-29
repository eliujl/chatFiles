# chatbot using Langchain, OpenAI (API key required),
# and existing Pinecone index (API key required).

# Import required libraries
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
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
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",
                 openai_api_key=OPENAI_API_KEY)

# Set up Pinecone env
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_API_ENV = os.environ['PINECONE_API_ENV']
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Load the pre-created Pinecone index.
# The index which has already be stored in pinecone.io as long-term memory
index_name = input(
    Fore.GREEN + "Enter your Pinecone index: " + Style.RESET_ALL)
if index_name in pinecone.list_indexes():
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
else:
    raise ValueError('''Cannot find the specified Pinecone index.
                     Create one in pinecone.io or using 
                     pinecone.create_index(
                        name=index_name, dimension=1536, metric="cosine", shards=1)''')

# Ask if the source document should be printed out.
# If yes, the ouptput could be very long
print_source = input(Fore.GREEN +
                     "You want to also print texts from one most relevant source document? y/n: "
                     + Style.RESET_ALL)

# number of sources (split-documents when ingesting files); default is 4
k = 20

# Set up retriever
retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 20})

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
        # Print the first (most relevant) source
        # (only the first 400 characters for brevity here) if needed
        if print_source.lower() == 'y':
            print(Fore.GREEN + "Source document content:\n" +
                  Style.RESET_ALL,
                  source[0].page_content[:400])

        # Update the chat history with the user input and system response
        chat_history.append((query, reply))


chatbot_loop()
