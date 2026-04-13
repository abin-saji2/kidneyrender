from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = load_model("kidney_cnn_model.h5")
classes = ['Cyst', 'Normal', 'Stone', 'Tumor']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((128,128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)
    result = classes[np.argmax(pred)]
    confidence = float(np.max(pred) * 100)

    return {"prediction": result, "confidence": confidence}