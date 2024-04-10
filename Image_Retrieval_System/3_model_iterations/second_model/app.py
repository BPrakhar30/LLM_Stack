import os
import json
import numpy as np
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

# Function for generating embeddings
def generate_embeddings(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# Function for calculating cosine similarities
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Loading descriptions
def load_descriptions(path="descriptions"):
    descriptions = {}
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), 'r') as f:
                data = json.load(f)
                description = data["choices"][0]["message"]["content"]
                descriptions[filename.replace('.json', '')] = description
    return descriptions

# Function to convert descriptions to embeddings using OpenAI
def convert_to_embeddings(descriptions):
    description_embeddings = {}
    for key, description in descriptions.items():
        embedding = generate_embeddings(description, model="text-embedding-3-large")
        description_embeddings[key] = embedding
    return description_embeddings

descriptions = load_descriptions(r"C:\Users\prakh\Desktop\CMU_EDU\Chima\dataset_google\descriptions")

# Converting all descriptions to embeddings
description_embeddings = convert_to_embeddings(descriptions)


def find_best_matches(query, descriptions, description_embeddings, top_n=1, threshold=0.4):
    query_embedding = generate_embeddings(query, model="text-embedding-3-large")
   
    scores = {}
    for key, embedding in description_embeddings.items():
        score = cosine_similarity(embedding, query_embedding)
        scores[key] = score
    
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)   
    filtered_scores = [(key, score) for key, score in sorted_scores if score >= threshold][:top_n]
    print(filtered_scores)
    
    if not filtered_scores:
        return [], []
    
    image_names, filtered_scores = zip(*filtered_scores)   
    return image_names, filtered_scores


st.title('Image Retrieval System')
query = st.text_input('Enter your search query:', '')
number_of_results = st.selectbox('Number of results to retrieve:', [1, 5, 10], index=0)

if st.button('Search'):
    if query:
        matches, scores = find_best_matches(query, descriptions, description_embeddings, top_n=number_of_results, threshold=0.25)
        
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