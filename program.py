import torch
import torch.nn as nn
import faiss
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import csv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained("efficientnet-b0")  # Use B0 for speed
model = model.to(device)
model.eval()

# Remove the classification layer to get feature embeddings
feature_extractor = nn.Sequential(*list(model.children())[:-1])
def extract_features(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.extract_features(image)  # Correct method to get features
        features = torch.mean(features, dim=[2, 3])  # Global average pooling
        features = features.squeeze().cpu().numpy()  # Convert to NumPy array

    return features

def show_images(img1_path, img2_path, title1="Test Image", title2="Similar Image"):
    img1 = cv2.imread(img1_path)[..., ::-1]  # Convert BGR to RGB
    img2 = cv2.imread(img2_path)[..., ::-1]  # Convert BGR to RGB

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(title2)
    plt.axis("off")

    plt.show()
    
from pathlib import Path

dataset_folder = Path("/home/sheerin/AnimalCLEF/Train/")
image_paths = [str(p) for p in dataset_folder.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".JPG"]]

#dataset_folder = "/home/sheerin/AnimalCLEF/images/"
#image_paths = glob.glob(f"{dataset_folder}/*.jpg")  # Change extension if needed

# Extract features for all training images
train_features = np.array([extract_features(img) for img in image_paths], dtype="float32")

# Build FAISS index
dimension = train_features.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance index
index.add(train_features)  # Add training features to FAISS

print(f"✅ FAISS index built with {index.ntotal} images.")
print(f"Found {len(image_paths)} images.")
if len(image_paths) == 0:
    raise ValueError("No images found in the dataset folder. Check the path and file extensions.")

test_image_paths = "/home/sheerin/AnimalCLEF/Test/"
test_image_paths = Path(test_image_paths)  # convert string to Path
imgs_test = [str(p) for p in test_image_paths.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".JPG"]]


print(imgs_test)
for img_path in imgs_test:
  #test_image_path = "/content/test.jpg"  # Replace with actual test image
  test_feature = extract_features(img_path).reshape(1, -1).astype("float32")
  print("✅ Test feature extracted.")
  k = 1  # Number of nearest neighbors to find
  D, I = index.search(test_feature, k)
  print("D,I", D,I)
  # Get the most similar image
  closest_image_path = image_paths[I[0][0]]  # Retrieve matched image path
  print(f"✅ Most similar image found: {closest_image_path}")
  show_images(img_path, closest_image_path)
  with open('/home/sheerin/AnimalCLEF/finaloutput.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    #writer.writerow(['image_id', 'identity'])
    label = closest_image_path if D < 50 else "new_individual"
    writer.writerow([img_path, D, label])

      