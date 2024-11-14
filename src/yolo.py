from ultralytics import YOLO
import cv2
import os

model_path = 'C:/Users/Sasa/Desktop/data/runs/content/runs/detect/train3/weights/best.pt'

model = YOLO(model=model_path)

images_dir = 'C:/Users/Sasa/Desktop/data/images3'

images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

# Create cropped images directory if it doesn't exist
cropped_dir = 'C:/Users/Sasa/Desktop/data/cropped_dir3'
os.makedirs(cropped_dir, exist_ok=True)

for image in images:
    img_path = os.path.join(images_dir, image)
    
    results = model.predict(source=img_path, save=True, conf=0.5)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Loop through the results and crop the region of interest
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates: [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop the image using the bounding box coordinates
            cropped_image = img[y1:y2, x1:x2]

            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

            # Check if the cropped_image is not empty
            if cropped_image.size == 0:
                print(f"Empty cropped image for {image}. Skipping...")
                continue
            
            # Save the cropped_image_rgb and get the last string of the filename
            filename = f"cropped_{image}"
            cv2.imwrite(os.path.join(cropped_dir, filename), cropped_image_rgb)

            print(f"Saved {filename}!")


