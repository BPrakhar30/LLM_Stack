import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import streamlit as st
import os
import faiss
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Generating the embeddings for text and query
def generate_embeddings(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   embedding = client.embeddings.create(input = [text], model=model).data[0].embedding
   return np.array(embedding)


# Loading the descriptions in JSON format
def load_descriptions(path="descriptions"):
    descriptions = {}
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), 'r') as f:
                data = json.load(f)
                description = data["choices"][0]["message"]["content"]
                descriptions[filename.replace('.json', '')] = description
    return descriptions

descriptions = load_descriptions(r"path_to_the_descriptions_in_JSON_format")

def create_faiss_index(embeddings):

    embeddings_matrix = np.vstack(embeddings)  
    d = embeddings_matrix.shape[1] 
    index = faiss.IndexFlatL2(d)  
    index.add(embeddings_matrix.astype('float32')) 
    return index


def convert_to_embeddings_faiss(descriptions):
    """
    Converting the descriptions to embeddings and creating a FAISS index
    """
    description_embeddings = []
    keys = []
    for key, description in descriptions.items():
        embedding = generate_embeddings(description, model="text-embedding-3-large")
        description_embeddings.append(np.array(embedding))
        keys.append(key)
    faiss_index = create_faiss_index(description_embeddings)
    return keys, faiss_index

keys, faiss_index = convert_to_embeddings_faiss(descriptions)

# Converting descriptions to embeddings using OpenAI
def convert_to_embeddings(descriptions):
    description_embeddings = {}
    for key, description in descriptions.items():
        embedding = generate_embeddings(description, model="text-embedding-3-large")
        description_embeddings[key] = embedding
    return description_embeddings

# Converting all descriptions to embeddings
description_embeddings = convert_to_embeddings(descriptions)

# Plotting the probability density function 
def log_matches_with_graph(distances, indices, keys, graph_filename="distance_graph.png"):
    all_distances = distances.flatten()
    mean, std = np.mean(all_distances), np.std(all_distances)
    x = np.linspace(min(all_distances), max(all_distances), 100)
    
    # Calculating probability density function 
    pdf = stats.norm.pdf(x, mean, std)
    
    # Plotting the bell curve
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, color='blue', linestyle='-', linewidth=2)
    plt.fill_between(x, pdf, color='blue', alpha=0.1)
    plt.title('Distance Distribution Bell Curve')
    plt.xlabel('Distance')
    plt.ylabel('Probability Density')
    plt.savefig(graph_filename)
    plt.close()
    print(f"Bell curve graph has been saved as {graph_filename}.")

    return (mean - 0.88*std)

def find_best_matches(query, keys, faiss_index, top_n):
    # Generating embeddings for the query
    query_embedding = generate_embeddings(query, model="text-embedding-3-large").reshape(1, -1).astype('float32')
    # Searching the FAISS index for the top_n closest matches
    distances, indices = faiss_index.search(query_embedding, top_n)
    threshold = log_matches_with_graph(distances, indices, keys)
    print(threshold)

    filtered_indices = []
    filtered_scores = []
    for i, distance in enumerate(distances[0]):
        if distance < threshold:  
            filtered_indices.append(indices[0][i])
            filtered_scores.append(distance)
    
    image_names = [keys[idx] for idx in filtered_indices]
    
    return image_names, filtered_scores

st.title('Image Retrieval System')

query = st.text_input('Enter your search query:', '')

number_of_results = st.selectbox('Number of results to retrieve:', [1, 5, 10], index=0)

if st.button('Search'):
    if query:
        matches, scores = find_best_matches(query, keys, faiss_index, top_n=number_of_results)
        print(matches, scores)
        
        if matches:
            if len(matches) < number_of_results:
                st.write(f"Only {len(matches)} images found with a similarity score above the threshold.")
            
            num_rows = (len(matches) + 3) // 4
            
            matches_in_rows = [matches[i:i+4] for i in range(0, len(matches), 4)]
            
            for row in matches_in_rows:
                cols = st.columns(4)
                for i, match in enumerate(row):
                    image_path_jpg = os.path.join('images', f'{match}.jpg')
                    image_path_png = os.path.join('images', f'{match}.png')
                    print(image_path_jpg)

                    if os.path.isfile(image_path_jpg):
                        image_path = image_path_jpg
                    elif os.path.isfile(image_path_png):
                        image_path = image_path_png
                    else:
                        st.error(f"Image file for '{match}' not found.")
                        continue
                    
                    with cols[i]:
                        st.image(image_path, caption=f'Result {i+1} for: "{query}"', use_column_width=True)
        else:
            st.write('No matching images found.')