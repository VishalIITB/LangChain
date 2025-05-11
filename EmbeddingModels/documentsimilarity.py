from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Initialize the embedding model
embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
)

# Documents
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# Query
query = 'tell me about Bumrah'

# Generate embeddings
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Calculate cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the most similar document
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"Query: {query}")
print(f"Most Similar Document: {documents[index]}")
print(f"Similarity Score: {score}")
