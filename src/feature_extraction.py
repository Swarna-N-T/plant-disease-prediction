import numpy as np
import cv2

def extract_features(img):
    img_uint8 = (img * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    return hist.flatten()

def generate_feature_matrix(data):
    X_features = np.array([extract_features(img) for img in data])
    return X_features
