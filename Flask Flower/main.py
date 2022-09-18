import keras
from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf


app = Flask(__name__)
upload_folder = "/home/Yasmine/PycharmProjects/Flask Flower/templates"

def predict( image_file):
    model = keras.models.load_model('/home/Yasmine/PycharmProjects/Flask Flower/my_model.h5')
    img = cv2.imread(image_file)
    img = cv2.resize(img, (224, 224))  # resize image to match model's expected sizing
    img = img.reshape(1, 224, 224, 3)
    img = img / 225
    img2 = tf.cast(img, tf.float32)
    pre = np.argmax(model.predict(img2))
    if pre == 0:
        bo = "daisy"
    if pre == 1:
        bo = "dandelion"
    if pre == 2:
        bo = "rose"
    if pre == 3:
        bo = "sunflower"
    if pre == 4:
        bo = "tulip"
    return bo

@app.route('/', methods=["GET", "POST"])
@app.route('/home', methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
      image_file = request.files["image"]
      if image_file:
          image_location = os.path.join(
              upload_folder,
              image_file.filename
          )
          image_file.save(image_location)
          bo = predict( image_location)

          return render_template("home.html", pre = predict( image_location))
    return render_template("home.html", pre = 0)

if __name__ =="__main__":
    app.run(port=12000, debug=True)


