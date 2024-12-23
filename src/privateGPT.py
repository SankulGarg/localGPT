from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
import os
import argparse
import time
from vectorstore.vectorStoreFactory import VectorStoreFactory as vectorStoreFactory
import util.documentParser as documentParser
import service.ingestService as ingestService
import util.PythonPropertyReader as pythonPropertyReader


model = pythonPropertyReader.get("MODEL")
embeddings_model_name = pythonPropertyReader.get("EMBEDDINGS_MODEL_NAME")
target_source_chunks = int(pythonPropertyReader.get('TARGET_SOURCE_CHUNKS'))
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
db = pythonPropertyReader.get("DB")

def main():
    # Parse the command line arguments
    args = parse_arguments()
    # Load the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vector_store_interface = vectorStoreFactory.get_vector_db(db, embeddings)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

    # The retriever should work with the embeddings and return the top-k documents relevant to the query
    retriever = vector_store_interface.db.as_retriever(search_kwargs={"k": target_source_chunks})
    
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = Ollama(model=model, callbacks=callbacks)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=retriever, return_source_documents=True)
    
    # Interactive questions and answers 
    def handle_query(query):
        if query == "/exit":
            return True  # Exit the loop

        elif query == "/clear":
            os.system("clear")

        elif query == "/ingest":
            documents = documentParser.process_documents(vector_store_interface.get_stored_files())
            print(f"Loaded {len(documents)} documents.")
            ingestService.ingest(vector_store_interface, documents)

        elif query.startswith("/ingest"):
            file_paths = query.split(" ")[1:]
            if not file_paths:
                print("No file paths provided.")
            else:
                for file_path in file_paths:
                    documentParser.load_documents(file_path)

        elif query.strip() != "":
            # Retrieve relevant documents based on the query and embeddings
            retrieved_docs = retriever.get_relevant_documents(query)
            

            # Get the answer from the chain, including the context and retrieved documents
            start = time.time()
            res = qa(query)  # Append the query and the context to the LLM
            end = time.time()
            
            print(f"({end-start:.2f}s) Time taken")
    

    while True:
        query = input("\nEnter a query: ")
        exit_flag = handle_query(query)
        if exit_flag:
            break


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                         help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
