import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the saved Keras model
model_path = "wildfire_model.keras"
model = load_model(model_path)

# Preprocess image function
def preprocess_image_pil(image):
    img = image.convert("RGB")
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Predict function
def predict_image_pil(image):
    img = preprocess_image_pil(image)
    pred = model.predict(img)
    print("Raw prediction:", pred)  # Debug print
    
    # Check prediction shape
    if pred.shape[-1] == 1:  # Binary classification
        score = pred[0][0]
        print(f"Prediction score: {score}")  # Debug print
        if score > 0.5:
            return f"Wildfire 🔥 (confidence: {score:.2%})"
        else:
            return f"No Wildfire ✅ (confidence: {(1-score):.2%})"
    else:  # Multi-class classification
        class_probs = pred[0]
        print(f"Class probabilities: {class_probs}")  # Debug print
        class_idx = np.argmax(class_probs)
        confidence = class_probs[class_idx]
        if class_idx == 1:
            return f"Wildfire 🔥 (confidence: {confidence:.2%})"
        else:
            return f"No Wildfire ✅ (confidence: {confidence:.2%})"

st.title("Forest Fire Prediction")
st.write("Upload an image to predict if it shows a wildfire.")

# Use session state to manage file uploader reset
if "reset_uploader" not in st.session_state:
    st.session_state.reset_uploader = False

def reset_uploader():
    st.session_state.reset_uploader = True

if st.session_state.reset_uploader:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="new_uploader")
    st.session_state.reset_uploader = False
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Predicting..."):
        result = predict_image_pil(image)
    st.markdown(
        f"<h2 style='color: green; font-size: 2.2em;'>Prediction: {result}</h2>",
        unsafe_allow_html=True
    )
    if st.button("Insert Another Image"):
        reset_uploader()
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()