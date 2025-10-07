import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from tkinter import Tk, filedialog
from PIL import Image

# ------------------------
# Load the saved Keras model
# ------------------------
model_path = "wildfire_model.keras"  # Change if your folder is elsewhere
model = load_model(model_path)
print("Model loaded successfully!")

# ------------------------
# Preprocess image function
# ------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)         # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (32, 32))     # Resize to model input
    img = img / 255.0                    # Normalize
    img = np.expand_dims(img, axis=0)    # Add batch dimension
    return img

# ------------------------
# Predict function
# ------------------------
def predict_image(image_path):
    img = preprocess_image(image_path)
    pred = model.predict(img)
    print("Raw prediction:", pred)  # Debug print
    
    # Get model summary to understand output shape
    model.summary()
    
    # Check prediction shape
    if pred.shape[-1] == 1:  # Binary classification
        score = pred[0][0]
        print(f"Prediction score: {score}")  # Debug print
        if score > 0.5:
            print(f"Prediction: Wildfire 🔥")
        else:
            print(f"Prediction: No Wildfire ✅")
    else:  # Multi-class classification
        class_probs = pred[0]
        print(f"Class probabilities: {class_probs}")  # Debug print
        class_idx = np.argmax(class_probs)
        if class_idx == 1:
            print(f"Prediction: Wildfire 🔥")
        else:
            print(f"Prediction: No Wildfire ✅")

# ------------------------
# GUI for selecting image
# ------------------------
def select_and_predict():
    # Hide main Tkinter window
    root = Tk()
    root.withdraw()

    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        # Show selected image
        img = Image.open(file_path)
        img.show()

        # Predict
        predict_image(file_path)

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    print("Select an image for wildfire prediction...")
    select_and_predict()
