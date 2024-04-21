from chromadb.config import Settings
import os
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY", "db")
# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)