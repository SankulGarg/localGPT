from langchain.vectorstores.pgvector import PGVector
from vectorstore.vectorStoreProvider import VectorStoreProvider
from repository.vectorInterface import VectorInterface
from configs.db.postgres_config import CONNECTION_STRING, COLLECTION_NAME
from repository.postgres import Postgres

class PostgresProvider(VectorStoreProvider):
    
    def __init__(self):
        self.db = None
    
    def get_db(self, embedding_function)-> VectorInterface:
        if self.db is None:
            PGVector.from_embeddings
            self.db = PGVector.from_existing_index(
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
                embedding=embedding_function)
        return Postgres(self.db)