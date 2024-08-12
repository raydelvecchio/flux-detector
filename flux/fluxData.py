import replicate
import os
import requests
from tqdm import tqdm
import random
import dotenv
dotenv.load_dotenv('../.env')

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

def generate_and_save_image():
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

if __name__ == "__main__":
    for _ in tqdm(range(1000), desc="Generating images"):
        generate_and_save_image()
