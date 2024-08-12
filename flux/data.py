from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException, ElementNotInteractableException
import requests
import time
import urllib.parse
import os

def download_images(query, limit, output_directory):
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

if __name__ == "__main__":
    download_images(query="woman presenting on stage", limit=1000, output_directory="google_images")