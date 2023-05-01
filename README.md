# chatFiles

## A small collection of tools for chatting with files

Some easy to make / easy to use tools for chatting with files. For most of them, as long as you set up the environment, enter the filename and needed API key(s), you can start to chat.

## Descriptions
1. 'usefulResources.md' has a short list of resources / tutorials / apps on chatting with files.
2. 'pdf_retrieve_qa.ipynb' is an iPython notebook for a PDF file. It uses the PDF file loader, the text splitter, and retriever/QA chain from Langchain, vector store from Chroma (which does not require any API key but data is only transient), and chat model from OpenAI (API key required). 
3. 'doc_retrieve_qa.ipynb' is an iPython notebook for a Word document. It uses the Word file loader, the text splitter and  retriever/QA chain from Langchain, vector store from Chroma (which does not require any API key but data is only transient), and chat model from OpenAI (API key required). 
4. 'pdf_qa_pinecone.ipynb' is an iPython notebook for a PDF file. It uses the PDF file loader, the text splitter, and QA chain from Langchain, vector store from Pinecone (which require an API key and data can be stored in pinecone.io), and chat model from OpenAI (API key required). 
5. 'doc_retrieve_qa_faiss.ipynb' is an iPython notebook for a Word document. It uses the Word file loader and text splitter from Langchain, vector store / retriever / QA chain from Langchain FAISS, and embeddings from OpenAI (API key required). 
6. 'qa_load_pinecone.ipynb' is an iPython notebook for Q&A with vectors preloaded to Pinecone. It does not load/ingest any file, but another app has done the ingestion/etc. and stored the vectors in a Pinecone index with an index_name. Then this file loads the Pinecone index (API key required), use Langchain's load_qa_chain and OpenAI (API key required) as LLM, and performs the Q&A.
7. 'docs_converseretrieve_pinecone.ipynb' is an iPython notebook for multiple Word document. It uses the Word file loader, the text splitter and  conversational retrieval chain from Langchain which enables chat history, vector store from Pinecone (API key required), and LLM from OpenAI (API key required). 
8. 'converse_load_pinecone.ipynb' is an iPython notebook for conversation with vectors preloaded to Pinecone. It does not load/ingest any file, but another app has done the ingestion/etc. and stored the vectors in a Pinecone index with an index_name. Then this file loads the Pinecone index (API key required), use Langchain's conversational retrieval chain and OpenAI (API key required) as LLM, and performs the conversation.
9. 'chatbot_load_pinecone.py' is a simple Python chatbot to chat with vectors preloaded to Pinecone. It does not load/ingest any file, but another app has done the ingestion/etc. and stored the vectors in a Pinecone index with an index_name. Then this file loads the Pinecone index (API key required), use Langchain's conversational retrieval chain and OpenAI (API key required) as LLM, and performs the conversation. See screenshot below.
10. 'chatbot_for_files.py' is a Python chatbot to chat with data. The data may be pre-ingested in vector stores such as Pinecone (online; with a Pinecone index_name) or Chroma (local), or ingested from one or more files (which could be Word documents, PDF files, and .txt files). If Pinecone is to be used, a Pinecone API key, environment information, and an index are required, otherwise Chroma is used. It  uses Langchain's conversational retrieval chain and OpenAI (API key required) as LLM, and performs the conversation. 

## Notes
The coding is straightforward, and can be readily modified for other purposes and to use other tools. For example, 'pdf_retrieve_qa.ipynb' and 'doc_retrieve_qa.ipynb' differ only by the loader, and you can change the loader to load other types of files. The text splitter, the retriever, vector store, and the chat model, can all be easily changed. The deliberate approach here is to maintain distinct files for various component combinations for simplicity of using each file, instead of having a single comprehensive file that encompasses these combinations with different settings. 

This is only a very small collection. Will try to add a few more later.

## Screenshot
![chatbot_load_pinecone](https://user-images.githubusercontent.com/8711788/235317845-a94a3122-1e56-4ef4-81d4-3fbc89910093.png)