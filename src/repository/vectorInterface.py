
from abc import ABC
from langchain.vectorstores.base import VectorStore
class VectorInterface(ABC):

    def __init__(self, db: VectorStore):
        self.db = db
    
    
    def get_stored_files(self)-> list[str]:
        #Implement the logic to retrieve the database using the embedding function.
        pass

    def persist(self, documents):
        #Implement the logic to check if the database exists.
        pass