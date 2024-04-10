# 0 - Image Dataset

## Dataset Information
The dataset has a collection of images across a wide range of categories such as sports, animals, social interactions, and digital artwork.

### Script Functionality - dataset_web_scraping.py
Automated the downloading of images from Google Search. It constructs a URL for the image search, fetches the results, and saves a specified number of images to a local directory.

Kaggle datasets were also used to collect some categories of data

### Categories Included in the Dataset

Adventure Sports, African Traditional Art, African Traditional Apparel, Analysis of a Pie Chart, Animals in the Wild, Animals Playing, Athletics, Car Accident, Celebrity Instagram Posts, Cricket Game, Formula One, Happy People, Indian Traditional Art, Indian Traditional Apparel, Maps of Different Regions in the USA with Analysis, Maps of Regions in Texas, People Chatting in Church, People in a Party, People in Public Transport, People Protesting, Sad People, Street Violence, Ted Talks with Person on Stage, Tweets on Twitter, Violent Crowd

# 1 - Images Tagging Script

## Overview
This script processes images and generates descriptive tags using the vision language model from OPENAI. It encodes images, communicates with an API to get the descriptions, and saves the results in a JSON format.

### About the script
**Image Encoding**: Each image in the specified directory is encoded into a base64 format, in this way text is represented as binary data.

**Tag Generation**: The script sends the encoded image to the OpenAI VLM with a prompt to generate descriptive tags. These tags include elements such as subjects, actions, symbols, text, and emotions observed in the image.

**Saving Output**: The generated tags and information from the model are saved as a JSON file. Each JSON file corresponds to its respective image.

# 2 - Image Tags

Three prompts were used to understand how the retrieval system performs with varied descriptions:

- The first prompt requested a description using keywords without filler words, focusing on main subjects, objects, context, and emotional content.
- The second prompt requested to generate a concise and accurate description within a 20-30 word limit.
- The third prompt asked for generating tags rather than full sentences, asking for various elements of the image, including mood and themes.

# 3 - Model Iteration 

## Basic Task of the model - How the Image Retrieval System Works
The system uses an LLM model to understand and compare the descriptions of the images and the query asked by the user.
The code reads descriptions from files in JSON format.
Each description is turned into an embedding.
Once a search query is entered, those too are converted into embeddings.
We then compare the query embedding with all the description embeddings to find the closest matches.
After finding the matches, we display them on the UI created with Streamlit.
If no matches are found, we inform the user that no images could be matched with the query.

### A. Initial Model Details

**Embedding model used** - 'all-MiniLM-L6-v2', a sentence transformer-based model
**Similarity measurement** - Cosine Similarity

### B. Second Model Details 

**Embedding model used** - "text-embedding-3-large", an OPEN AI LLM model
**Similarity measurement** - Cosine Similarity based on the threshold so that only matches greater than the set threshold are retrieved

### C. Third Model Details

**Embedding model used** - "text-embedding-3-large", an OPEN AI LLM model
**Similarity measurement** - FAISS L2 norm for similarity search based on the threshold (calculated using the probability density function of distance)

# 4 - Tests

All tests were performed on the third model
These tests focus on testing three types of descriptions generated as mentioned in **2 - Image Tags**.
