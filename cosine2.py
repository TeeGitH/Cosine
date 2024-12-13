import numpy as np
from openai import OpenAI
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dotenv import load_dotenv
import os

def get_embedding(text, client):
    """Get embedding for a given text using OpenAI API"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(word1, word2, client):
    """Calculate cosine similarity between two words"""
    # Get embeddings for both words
    embedding1 = get_embedding(word1, client)
    embedding2 = get_embedding(word2, client)
    
    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    
    return similarity

def update_plot(words, similarities):
    """Update the plot with new data"""
    plt.clf()  # Clear the current figure
    bars = plt.bar(words, similarities)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)  # Add reference line at y=1
    
    # Customize the plot
    plt.ylim(-1, 1.1)
    plt.title(f"Similarity to target word: '{target_word}'")
    plt.xlabel("Guessed Words")
    plt.ylabel("Cosine Similarity")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Small pause to update the plot

def interactive_similarity_check(target_word, api_key):
    """Interactive function to continuously check similarity with a target word"""
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Initialize plot
    plt.figure(figsize=(10, 6))
    words = []
    similarities = []
    
    print(f"\nTarget word is set to: '{target_word}'")
    print("Enter words to compare (type 'exit' to quit)")
    
    while True:
        user_input = input("\nEnter a word: ").strip().lower()
        
        if user_input == 'exit':
            print("Goodbye!")
            break
        
        if user_input:
            try:
                similarity = cosine_similarity(target_word, user_input, client)
                print(f"Similarity between '{target_word}' and '{user_input}': {similarity:.4f}")
                
                # Update data for plotting
                words.append(user_input)
                similarities.append(similarity)
                
                # Update the plot
                update_plot(words, similarities)
                
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your .env file")
        
    target_word = "mango"  # You can change this to any word you want
    
    plt.ion()  # Turn on interactive plotting
    interactive_similarity_check(target_word, api_key)
    plt.ioff()  # Turn off interactive plotting when done
    plt.show()  # Keep the final plot window open
