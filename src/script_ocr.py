from paddleocr import PaddleOCR, draw_ocr
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load and preprocess image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_mean = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return adaptive_mean
    

# Perform OCR and return the results
def perform_ocr(image):
    ocr_results = ocr.ocr(image, cls=True)
    
    if not ocr_results or not ocr_results[0]:
        print("No ocr results found.")
        return [], [], []
    
    # Extract bounding boxes, text, and confidence scores
    boxes = [line[0] for line in ocr_results[0]]
    texts = [line[1][0] for line in ocr_results[0]]
    scores = [line[1][1] for line in ocr_results[0]]
    return boxes, texts, scores

# Check for text match and update the DataFrame
def check_and_update_match(texts, value_to_match):
    for text in texts:
        if value_to_match.lower() in text.lower():
            return True
    return False

import numpy as np

# Draw bounding boxes for matched text
def draw_matched_text(font, original_image, boxes, texts, scores, value_to_match):
    matched_boxes, matched_texts, matched_scores = [], [], []
    for i, text in enumerate(texts):
        if value_to_match.lower() in text.lower():
            matched_boxes.append(boxes[i])
            matched_texts.append(text)
            matched_scores.append(scores[i])
            break

    # Draw the bounding boxes on the original image
    image_with_matched_ocr = draw_ocr(
        Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)),
        matched_boxes,
        matched_texts,
        matched_scores,
        font_path=font
    )
    return cv2.cvtColor(np.array(image_with_matched_ocr), cv2.COLOR_RGB2BGR)

# Main function to process images in the folder
def process_images(folder_path, font_path, csv_file):
    df = pd.read_csv(csv_file)
    df['Value'] = df['Value'].astype(str)
    df['Matched'] = False

    # Iterate through image files in the folder
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        print(f"\nProcessing {filename}...")
        try:
            #filename_index = int(filename.split('_')[1].split('.')[0])
            filename_index = int(filename.split('.')[0])
        except (IndexError, ValueError):
            print(f"Invalid filename format for {filename}. Skipping...")
            continue

        if filename_index >= len(df):
            print(f"Index {filename_index} out of range in CSV. Skipping...")
            continue

        value_to_match = df.loc[filename_index, 'Value']

        # Load and preprocess the image
        image = load_image(os.path.join(folder_path, filename))

        # Perform OCR
        boxes, texts, scores = perform_ocr(image)

        if not texts:
            print(f"No text detected in {filename}. Skipping...")
            continue

        # Update DataFrame if match is found
        if check_and_update_match(texts, value_to_match):
            df.at[filename_index, 'Matched'] = True
            print(f"Match found for '{value_to_match}' in {filename}.")

        # Draw the matched text
        #processed_image = draw_matched_text(font_path, image, boxes, texts, scores, value_to_match)

        # Display the processed image
        #plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        #plt.axis('off')
        #plt.show()

    # Optionally save the updated CSV
    output_csv_path = os.path.join(os.path.dirname(csv_file), 'Updated_Readings3.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"\nUpdated CSV saved to: {output_csv_path}")
    return output_csv_path

def check_accuracy(csv_file):
    df = pd.read_csv(csv_file)
    total = len(df)
    matched = len(df[df['Matched'] == True])
    accuracy = (matched / total) * 100
    print(f"\nTotal images: {total}")
    print(f"Matched images: {matched}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    font_path = 'C:/Users/Sasa/Desktop/data/PaddleOCR/doc/fonts/latin.ttf'
    images_folder = 'C:/Users/Sasa/Desktop/data/cropped_dir3'
    csv_file = 'C:/Users/Sasa/Desktop/data/Readings3.csv'
    
    output_csv = process_images(images_folder, font_path, csv_file)
    check_accuracy(output_csv)