from vectorstore.chromaDBProvider import ChromaDBProvider
from vectorstore.postgresProvider import PostgresProvider
from repository.vectorInterface import VectorInterface

class VectorStoreFactory:
    
    db=None

    # def get_scalar_db(dbType)
    @staticmethod
    def get_vector_db(db_type, embeddings_model) -> VectorInterface:
        print(f"Databse: {db_type}")
        
        if VectorStoreFactory.db is not None:
            return VectorStoreFactory.db
        elif db_type == "chroma":
            VectorStoreFactory.db = ChromaDBProvider().get_db(embeddings_model)
        elif db_type == "postgres":
            VectorStoreFactory.db = PostgresProvider().get_db(embeddings_model)
        else:
            raise ValueError("Invalid database type")
        # Return the vector database connection object
        return VectorStoreFactory.db

