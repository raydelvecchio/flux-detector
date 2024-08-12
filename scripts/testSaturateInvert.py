import cv2
import numpy as np

def increase_saturation_with_inversion(image_path, saturation_factor=2.0):
    img = cv2.imread(image_path)
    inverted_img = cv2.bitwise_not(img)
    hsv = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation_factor, 0, 255)
    saturated_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return saturated_img

if __name__ == "__main__":
    input_image_path = "fa3b6f8464ec6186.jpg"
    output_image_path = input_image_path.rsplit('.', 1)[0] + "_INVERTED_SATURATED.jpg"

    result = increase_saturation_with_inversion(input_image_path, saturation_factor=500.0)
    cv2.imwrite(output_image_path, result)
