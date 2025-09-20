
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import IMAGE_SIZE, DATASET_PATH

def load_and_preprocess_images(dataset_path=DATASET_PATH, image_size=IMAGE_SIZE):
    data = []
    labels = []
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        for img in os.listdir(category_path):
            img_path = os.path.join(category_path, img)
            try:
                img_arr = cv2.imread(img_path)
                img_arr = cv2.resize(img_arr, (image_size, image_size))
                data.append(img_arr)
                labels.append(category)
            except:
                pass
    data = np.array(data) / 255.0
    labels = np.array(labels)
    return data, labels

def encode_labels(labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    return labels_encoded, le

def split_data(data, labels_encoded, test_size=0.2, random_state=42):
    return train_test_split(data, labels_encoded, test_size=test_size, random_state=random_state, stratify=labels_encoded)
