import streamlit as st
import os
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initializing the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Loading the descriptions from JSON files
def load_descriptions(path="descriptions"):
    descriptions = {}
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), 'r') as file:
                data = json.load(file)
                # Key: filename without extension, Value: description
                descriptions[filename[:-5]] = data["choices"][0]["message"]["content"]
    return descriptions

# Converting descriptions to embeddings
def convert_to_embeddings(descriptions):
    description_list = list(descriptions.values())
    return model.encode(description_list, convert_to_tensor=True)

# Finding the best matches for a query
def find_best_matches(query, descriptions, embeddings, top_n=1, threshold=0.0):
    query_embedding = model.encode(query, convert_to_tensor=True).to('cpu')
    similarities = util.pytorch_cos_sim(query_embedding, embeddings.to('cpu'))[0].cpu().numpy()
    
    # Sorting the indices of descriptions based on similarity scores
    sorted_indices = np.argsort(similarities)[::-1][:top_n]
    sorted_scores = similarities[sorted_indices]
    
    filtered_results = [(i, score) for i, score in zip(sorted_indices, sorted_scores) if score >= threshold]
    if not filtered_results:
        return [], []
    
    indices, scores = zip(*filtered_results)
    image_names = [list(descriptions.keys())[i] for i in indices]
    return image_names, scores

# Loadign and converting the descriptions into embeddings
descriptions = load_descriptions("path_to_the_descriptions_in_JSON_format")
description_embeddings = convert_to_embeddings(descriptions)

# Setting  up streamlit UI 
st.title('Image Retrieval System')
query = st.text_input('Enter your search query:', '')
number_of_results = st.selectbox('Number of results to retrieve:', [1, 5, 10], index=0)

if st.button('Search') and query:
    matches, similarities = find_best_matches(query, descriptions, description_embeddings, top_n=number_of_results)
    
    if matches:
        # We will display a message if fewer images than requested are found
        if len(matches) < number_of_results:
            st.write(f"Only {len(matches)} images found with a similarity score above the threshold.")
        
        # Displaying images in rows of 4
        for i in range(0, len(matches), 4):
            cols = st.columns(4)
            for col, match in zip(cols, matches[i:i+4]):
                image_path = next((os.path.join('images', f'{match}.{ext}') for ext in ['jpg', 'png'] if os.path.isfile(os.path.join('images', f'{match}.{ext}'))), None)
                if image_path:
                    col.image(image_path, caption=f'Result for: "{query}"', use_column_width=True)
                else:
                    col.error(f"Image file for '{match}' not found.")
    else:
        st.write('No matching images found.')
