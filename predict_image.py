import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from tkinter import Tk, filedialog
from PIL import Image

model_path = "wildfire_model.keras"  
model = load_model(model_path)
print("Model loaded successfully!")



def preprocess_image(image_path):
    img = cv2.imread(image_path)         
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (32, 32))     
    img = img / 255.0                    
    img = np.expand_dims(img, axis=0)    
    return img


def predict_image(image_path):
    img = preprocess_image(image_path)
    pred = model.predict(img)
    print("Raw prediction:", pred)  

    model.summary()
    
    if pred.shape[-1] == 1:  
        score = pred[0][0]
        print(f"Prediction score: {score}")  
        if score > 0.5:
            print(f"Prediction: Wildfire 🔥")
        else:
            print(f"Prediction: No Wildfire ✅")
    else:
        class_probs = pred[0]
        print(f"Class probabilities: {class_probs}")  
        class_idx = np.argmax(class_probs)
        if class_idx == 1:
            print(f"Prediction: Wildfire 🔥")
        else:
            print(f"Prediction: No Wildfire ✅")


def select_and_predict():

    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        img = Image.open(file_path)
        img.show()

        predict_image(file_path)


if __name__ == "__main__":
    print("Select an image for wildfire prediction...")
    select_and_predict()
