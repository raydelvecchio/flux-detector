from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import shutil
import random
import requests
import time
import urllib.parse
import os
import replicate
import os
import requests
from tqdm import tqdm
import dotenv
dotenv.load_dotenv('../.env')

def download_images_from_google(query, limit, output_directory):
    driver = webdriver.Chrome()
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://www.google.com/search?as_st=y&as_q={encoded_query}&imgsz=l&imgar=w&imgtype=photo&tbm=isch"
    driver.get(search_url)

    count = 0
    last_height = driver.execute_script("return document.body.scrollHeight")

    while count < limit:
        # Find all clickable image elements
        image_elements = driver.find_elements(By.CSS_SELECTOR, "div.eA0Zlc")  # DESCRIPTION: classname of the top level div for each image thumbnail!
        
        for img_div in image_elements[count:]:  # Start from where we left off
            if count >= limit:
                break
            
            try:
                img_div.click()
                
                # Wait for the full-size image to be present
                full_img = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "img.sFlh5c.pT0Scc.iPVvYb"))  # DESCRIPTION: this is the classname of the element (after clicking thumbnail) that has src set to the original large image!
                )
                
                img_url = full_img.get_attribute('src')
                if img_url and img_url.startswith('http'):
                    response = requests.get(img_url, stream=True)
                    if response.status_code == 200:
                        img_format = img_url.split('.')[-1].split('?')[0]
                        if img_format.lower() not in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                            img_format = 'jpg'
                        img_path = os.path.join(output_directory, f"image_{count}.{img_format}")
                        with open(img_path, 'wb') as f:
                            f.write(response.content)
                        count += 1
                        print(f"Successfully downloaded image {count}: {img_url}")
                
                # Close the full-size image view
                driver.execute_script("window.history.go(-1)")
                time.sleep(1)  # Wait for the search results to reload
            
            except Exception as e:
                print(f"Error downloading image: {e}")
                continue

        # Scroll down to load more images
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        if new_height == last_height:
            print("Reached the end of the page, no more images to load.")
            break
        
        last_height = new_height

    print(f"\nDownload process completed. Total images downloaded: {count}")
    driver.quit()

def save_replicate_output(output, folder="replicate_flux"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for url in output:
        response = requests.get(url)
        if response.status_code == 200:
            filename = f"{folder}/{os.urandom(8).hex()}.jpg"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Saved {filename}")
        else:
            print(f"Failed to download image from {url}")

def generate_and_save_image(base_prompt: str, descriptors: list[str]):
    selected_descriptors = random.sample(descriptors, 2)
    full_prompt = f"{base_prompt}, {', '.join(selected_descriptors)}"

    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": full_prompt,
            "num_outputs": 1,
            "aspect_ratio": "16:9",
            "output_format": "jpg",
            "output_quality": 100
        }
    )
    save_replicate_output(output)
    return output

def form_women_presenting_dataset():
    # Create the main dataset directory
    dataset_dir = "women_speaking_data"
    os.makedirs(dataset_dir, exist_ok=True)

    # Create subdirectories for train, test, and validation
    for split in ["train", "test", "validation"]:
        for label in ["REAL", "FAKE"]:
            os.makedirs(os.path.join(dataset_dir, split, label), exist_ok=True)

    # Collect all real and fake image paths
    real_images = [os.path.join("google_images", f) for f in os.listdir("google_images") if f.endswith(('.jpg', '.jpeg', '.png'))]
    fake_images = [os.path.join("replicate_flux", f) for f in os.listdir("replicate_flux") if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Function to split data without sklearn
    def custom_split(data, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
        random.shuffle(data)
        n = len(data)
        train_end = int(n * train_ratio)
        test_end = train_end + int(n * test_ratio)
        return data[:train_end], data[train_end:test_end], data[test_end:]

    # Split the datasets
    real_train, real_test, real_val = custom_split(real_images)
    fake_train, fake_test, fake_val = custom_split(fake_images)

    # Function to copy images to their respective directories
    def copy_images(images, split, label):
        for img in images:
            dest = os.path.join(dataset_dir, split, label, os.path.basename(img))
            shutil.copy(img, dest)

    # Copy images to their respective directories
    copy_images(real_train, "train", "REAL")
    copy_images(real_test, "test", "REAL")
    copy_images(real_val, "validation", "REAL")

    copy_images(fake_train, "train", "FAKE")
    copy_images(fake_test, "test", "FAKE")
    copy_images(fake_val, "validation", "FAKE")

    print("Dataset creation completed.")
    print(f"Train set: {len(real_train) + len(fake_train)} images")
    print(f"Test set: {len(real_test) + len(fake_test)} images")
    print(f"Validation set: {len(real_val) + len(fake_val)} images")

if __name__ == "__main__":
    # download_images(query="woman presenting on stage", limit=1000, output_directory="google_images")

    base_prompt = "woman presenting on stage"
    descriptors = [
        "close-up shot",
        "wide angle view",
        "from the audience perspective",
        "side view",
        "with a panel of experts",
        "holding a microphone",
        "gesturing with hands",
        "in front of a large screen",
        "with audience visible",
        "under dramatic lighting",
        "wearing a business suit",
        "with a confident pose",
        "mid-speech",
        "answering questions",
        "walking across the stage"
    ]
    # for _ in tqdm(range(1000), desc="Generating images"):
    #     generate_and_save_image(base_prompt, descriptors)

    form_women_presenting_dataset()
