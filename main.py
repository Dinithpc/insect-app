import PIL
from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from azure.storage.blob import BlobServiceClient
import h5py
app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
global_remedy = ""

connection_string='DefaultEndpointsProtocol=https;AccountName=insectpredct;AccountKey=N9IR0EUF53hmrw3hy5YYVLDTng1GwV5iHWmJqTNM6ql06KM5Y02+8pbd0Cnd70BNBC9EGhkXOFYZ+ASt+56kJg==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client("savedmodels")
blob_client = container_client.get_blob_client("Insect_bite_model.h5")
downloader = blob_client.download_blob(0)

with BytesIO() as f:
    downloader.readinto(f)
    with h5py.File(f, 'r') as h5file:
        global MODEL
        MODEL = tf.keras.models.load_model(h5file)
        MODEL.summary()

CLASS_NAMES = ['ants', 'bed_bugs', 'mosquito', 'no_bites', 'ticks']


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(PIL.Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    try:

        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        if predicted_class == "no_bites":
            return {
                'Message': 'Unknown Bite'
            }
        else:
            return {
                'Insect': predicted_class,
                'Message': 'Apply ' + global_remedy + ' onto the affected area.',
                'confidence': "{0:.0%}".format(float(confidence))
            }
    except Exception as e:
        error = "Error"
        return {
            'response': error
        }


@app.post("/allergy")
async def allergy(allergies: str):
    global global_remedy
    if allergies == "Redness":
        remedy = "Citronella Oil"
        global_remedy = remedy
        return {"message": "Allergy saved successfully"}
    elif allergies == "Itching":
        remedy = "Ice/Aloe/Honey/Olive oil"
        global_remedy = remedy
        return {"message": "Allergy saved successfully"}
    elif allergies == "Swelling":
        remedy = "Ice"
        global_remedy = remedy
        return {"message": "Allergy saved successfully"}
    elif allergies == "Pain":
        remedy = "Ice/Honey"
        global_remedy = remedy
        return {"message": "Allergy saved successfully"}
    elif allergies == "Heating":
        remedy = "Ice"
        global_remedy = remedy
        return {"message": "Allergy saved successfully"}
    else:
        return {"message": "Allergy type does not exist"}


if __name__ == "__main__":
    uvicorn.run(app)
