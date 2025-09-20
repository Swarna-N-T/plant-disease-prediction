import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image as PILImage
import gradio as gr

def load_cnn_model(model_path):
    return load_model(model_path)

def predict_disease_ui(image, cnn, label_encoder):
    img = np.array(image)
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_cv2, (64, 64))
    img_resized = img_resized / 255.0
    img_input = np.expand_dims(img_resized, axis=0)
    predictions = cnn.predict(img_input)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    class_name = label_encoder.inverse_transform([predicted_class])[0]
    top_indices = np.argsort(predictions[0])[::-1][:3]
    top_results = "\n".join(
        [f"{i+1}. {label_encoder.inverse_transform([idx])[0]} - {predictions[0][idx]:.2%}"
         for i, idx in enumerate(top_indices)]
    )
    result_text = (
        f"Class: {class_name}\n"
        f"Confidence: {confidence:.2%}\n\n"
        f"Top Predictions:\n{top_results}"
    )
    return result_text

def launch_gradio_ui(cnn, label_encoder):
    gr.Interface(
        fn=lambda image: predict_disease_ui(image, cnn, label_encoder),
        inputs=gr.Image(type="pil", label="Upload Leaf Image"),
        outputs=gr.Textbox(label="Prediction Results"),
        title="🌿 Plant Leaf Disease Predictor",
        description="Upload an image of a plant leaf. The model will detect possible diseases and show the top predictions."
    ).launch(share=True)
