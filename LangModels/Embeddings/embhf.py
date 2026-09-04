from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "This is a sample text to generate embeddings."

vector = embeddings.embed_query(text)

print("Generated Embedding Vector:")
print(vector)