import numpy as np
from openai import OpenAI
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Rest of your imports and functions remain the same...

def interactive_similarity_check(target_word, api_key):
    client = OpenAI(api_key=api_key)
    similarities = []
    words = []
    
    def on_click(event):
        if event.inaxes and event.button == 1:  # Left click
            word = input("Enter a word to compare with '" + target_word + "': ")
            response1 = client.embeddings.create(input=target_word, model="text-embedding-ada-002")
            response2 = client.embeddings.create(input=word, model="text-embedding-ada-002")
            
            embedding1 = response1.data[0].embedding
            embedding2 = response2.data[0].embedding
            
            similarity = np.dot(embedding1, embedding2)/(norm(embedding1)*norm(embedding2))
            similarities.append(similarity)
            words.append(word)
            
            # Print the similarity score
            print(f"\nSimilarity between '{target_word}' and '{word}': {similarity:.4f}")
            
            plt.clf()
            plt.bar(words, similarities)
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.title(f"Cosine Similarity with '{target_word}'")
            plt.draw()
    
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.title("Click anywhere to start comparing words")

if __name__ == "__main__":
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your .env file")
        
    target_word = "million"  # You can change this to any word you want
    
    plt.ion()  # Turn on interactive plotting
    interactive_similarity_check(target_word, api_key)
    plt.ioff()  # Turn off interactive plotting when done
    plt.show()  # Keep the final plot window open
