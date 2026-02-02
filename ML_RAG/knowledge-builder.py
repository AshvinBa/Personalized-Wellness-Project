import os
from langchain_chroma import Chroma

from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings



KNOWLEDGE_SOURCE_DIRECTORY = "knowledge_base"

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chroma = Chroma(
    persist_directory="./chroma",
    collection_name="GENERAL_GUIDELINES",
    embedding_function=embedding_function
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

documents = []

for file_name in os.listdir(KNOWLEDGE_SOURCE_DIRECTORY):
    file_path = os.path.join(KNOWLEDGE_SOURCE_DIRECTORY, file_name)

    with open(file_path, "rb") as file:
        contents = file.read().decode("utf-8", errors="ignore")

    chunks = text_splitter.create_documents(
        [contents],
        metadatas=[{"source": file_name}]
    )

    documents.extend(chunks)

BATCH_SIZE = 20
for i in range(0, len(documents), BATCH_SIZE):
    chroma.add_documents(documents[i:i + BATCH_SIZE])
    print(f"Embedded {i + BATCH_SIZE} / {len(documents)}")

print("✅ Knowledge base built successfully!")
