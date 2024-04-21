from repository.vectorInterface import VectorInterface


def ingest(vector_interface: VectorInterface, documents):
   vector_interface.persist(documents)


