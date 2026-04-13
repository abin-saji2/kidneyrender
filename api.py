from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io

app = FastAPI()

# 🔥 Lazy load model (IMPORTANT FIX)
model = None

def get_model():
    global model
    if model is None:
        from tensorflow.keras.models import load_model
        model = load_model("kidney_cnn_model.h5")
    return model

# Class labels
classes = ['Cyst', 'Normal', 'Stone', 'Tumor']

@app.get("/")
def home():
    return {"message": "Kidney API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = image.resize((128, 128))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Load model safely
        model = get_model()

        # Predict
        pred = model.predict(image)
        result = classes[np.argmax(pred)]
        confidence = float(np.max(pred) * 100)

        return {
            "prediction": result,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}