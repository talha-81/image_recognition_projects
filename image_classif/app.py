import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from typing import Optional

# Set page configuration
st.set_page_config(
    page_title="Pet Classifier AI",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Professional Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: white;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_classification_model() -> Optional[tf.keras.Model]:
    """
    Loads the pre-trained Keras model for image classification.

    Returns:
        tf.keras.Model: The loaded Keras model, or None if loading fails.
    """
    try:
        model = tf.keras.models.load_model("animal.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Prepares the image for model prediction by resizing and normalizing.

    Args:
        img (PIL.Image.Image): The input image.

    Returns:
        np.ndarray: The preprocessed image array ready for prediction.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((64, 64))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def main():
    # Sidebar for About/Info
    with st.sidebar:
        st.markdown("### Model Details")
        st.markdown("- **Input**: 64x64 RGB Images")
        st.markdown("- **Output**: Binary Classification (Cat vs. Dog)")
        st.markdown("---")

    # Main Content
    st.title("🐾 Pet Classifier AI")
    st.markdown("### Classify between Cats and Dogs instantly!")
    st.write("Upload an image of your pet, and our AI will analyze it.")

    model = load_classification_model()

    if model is not None:
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            try:
                # Display the uploaded image
                image = Image.open(uploaded_file)

                col1, col2 = st.columns([1, 1], gap="medium")

                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                with col2:
                    st.write("#### Analysis Result")
                    with st.spinner("Analyzing image features..."):
                        # Preprocess and Predict
                        processed_img = preprocess_image(image)
                        prediction = model.predict(processed_img)[0][0]

                        # Determine Label and Confidence
                        label = "Dog" if prediction > 0.5 else "Cat"
                        confidence = prediction if label == "Dog" else 1 - prediction

                        # Display Metrics
                        st.metric(
                            label="Prediction",
                            value=label,
                            delta=f"{confidence:.1%} Confidence",
                        )

                        # Visual Indicator
                        if label == "Dog":
                            st.success("It's a Dog! 🐶")
                        else:
                            st.error(
                                "It's a Cat! 🐱"
                            )  # using error/red for Cat just for color differentiation

            except Exception as e:
                st.error(f"Error processing image: {e}")

    st.divider()


if __name__ == "__main__":
    main()
