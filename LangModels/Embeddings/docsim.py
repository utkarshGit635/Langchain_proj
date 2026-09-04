from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv() 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "The old lighthouse blinked steadily across the quiet harbor.",
    "A curious cat watched raindrops race down the window.",
    "Bright kites floated above the park on a windy afternoon.",
    "The baker placed fresh cinnamon rolls behind the glass counter.",
    "An unexpected postcard arrived from a village near the mountains."
]

query = "Tell about postcard from the mountains."

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
print("Similarity Scores:")
print(scores)

# Get the index of the document with the highest similarity score
most_similar_index = np.argmax(scores)

print("Index of the most similar document:")
print(most_similar_index)
