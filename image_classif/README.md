# 🐾 Pet Classifier AI

A professional, deep-learning powered web application built with **Streamlit** and **TensorFlow** that classifies images of Cats and Dogs.

Confidence Scores**: Displays how certain the AI is about its prediction.

## 🛠️ Installation

1. **Navigate to the project directory:**

   ```bash
   cd image_classif
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Windows (Command Prompt):
   venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Run the application

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application file.
- `animal.h5`: Pre-trained Keras model (HDF5 format).
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation.

## Model Details

- **Input Shape**: 64x64 RGB images.
- **Preprocessing**: 1/255.0 normalization.
- **Classes**: Binary Classification (Cat vs. Dog).
