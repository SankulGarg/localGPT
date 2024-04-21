from langchain.vectorstores.base import VectorStore
from repository.vectorInterface import VectorInterface
class Postgres(VectorInterface):

    def __init__(self, db: VectorStore):
        self.db = db
    
    def get_stored_files(self)-> list[str]:
        if self.db is None or self.db.get() is None or self.db.get()['metadatas'] is None or len(self.db.get()['metadatas']) == 0:
            return []
        ingested_files = []
        for metadata in self.db.get()['metadatas']:
            print("metadata", metadata)
            if (metadata is not None):
                ingested_files.append(metadata['source'])
            print("Ingested files: ", ingested_files)
        return ingested_files
        

    def persist(self, documents):
        print("Persisting", len(documents), "to vectorstore")
        self.db.add_documents(documents)
        
    