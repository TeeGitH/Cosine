import numpy as np
from openai import OpenAI
from numpy.linalg import norm
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def get_embedding(text, client):
    """Get embedding for a given text using OpenAI API"""
    response = client.embeddings.create(
        model="text-embedding-3-small",  # or "text-embedding-ada-002" for older version
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(word1, word2, api_key):
    """Calculate cosine similarity between two words"""
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Get embeddings for both words
    embedding1 = get_embedding(word1, client)
    embedding2 = get_embedding(word2, client)
    
    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    
    return similarity

# Example usage
if __name__ == "__main__":
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your .env file")
    
    word1 = "million"
    word2 = "lion"
    
    similarity = cosine_similarity(word1, word2, api_key)
    print(f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}")
