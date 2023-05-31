import base64
import cv2
import numpy as np
import io
from PIL import Image  # Python Imaging Library
from tensorflow.keras.models import load_model
from flask.json import jsonify

model = load_model("models/guava_model.h5" , compile=False) #################### compile

diseaseNames = {
    "[0]": "Healthy",
    "[1]": "Mummification",
    "[2]": "Red_Rust",
}


def guava(imageString):

    image = base64.b64decode((imageString))  # Decode base64 to bytes
    image = np.array(Image.open(io.BytesIO(image)))  # open decoded image (bytes) as NP array

    input_im = cv2.resize(image, (512 , 512), interpolation=cv2.INTER_LINEAR) # resize image
    input_im = input_im / 255.  # normalize pixel value to 0 & 1
    input_im = input_im.reshape(1, 512 , 512, 3)  # batch_size , height , width , color
    pred = model.predict(input_im, 1, verbose=0)

    score = pred.max()
    score = score * 100

    res = np.argmax(pred, axis=1)
    disease = diseaseNames[str(res)]

    return jsonify({
        "diseaseName": disease,
        "conf": score,
    })