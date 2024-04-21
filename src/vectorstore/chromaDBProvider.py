from langchain.vectorstores.chroma import Chroma
from vectorstore.vectorStoreProvider import VectorStoreProvider
from repository.chromaDb import ChromaDB
from repository.vectorInterface import VectorInterface
from configs.db.chromaDB_config import PERSIST_DIRECTORY


class ChromaDBProvider(VectorStoreProvider):
    def __init__(self):
        self.db = None
    
    def get_db(self, embedding_function)-> VectorInterface:
        print(f"ChromaDB: {embedding_function}")
        if (self.db is None):
            self.db = Chroma.from_texts(["default"], embedding_function, persist_directory=PERSIST_DIRECTORY)
        return ChromaDB(self.db)
    