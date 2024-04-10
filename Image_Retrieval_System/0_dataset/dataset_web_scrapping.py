import os
import requests
from bs4 import BeautifulSoup

def download_images(search_term, number_of_images):
    # Constructing the Google search URL for images
    search_url = f"https://www.google.com/search?hl=en&tbm=isch&q={search_term.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    # Fetching the page and parsing it
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    images = [img['src'] if img['src'].startswith('http') else f"https://www.google.com{img['src']}" for img in soup.find_all('img')]

    directory = search_term.replace(' ', '_')
    os.makedirs(directory, exist_ok=True)

    # Downloading and saving the specified number of images
    for i, img_url in enumerate(images[:number_of_images]):
        try:
            img_data = requests.get(img_url, headers=headers).content
            with open(os.path.join(directory, f"{i+1}.jpg"), 'wb') as file:
                file.write(img_data)
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")

# Function call to download 10 images of given category
download_images("indian traditional apparel", 10)
