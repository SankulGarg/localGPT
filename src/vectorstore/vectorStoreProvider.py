from abc import ABC
from repository.vectorInterface import VectorInterface

class VectorStoreProvider(ABC):

    def get_db(self, embedding_function)-> VectorInterface:
        #Implement the logic to retrieve the database using the embedding function.
        pass
