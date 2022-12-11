# Import libraries
import tensorflow as tf
import numpy as np
from flask import Flask, request
import Utils

app = Flask(__name__)

model_path = 'output/cnn-model.h5'
input_image_path = 'output/api_input.jpg'
ml_model = Utils.load_model(model_path)
img_height = Utils.img_height
img_width = Utils.img_width
class_names = ["driving_license", "others", "social_security"]

@app.route("/get-image-class", methods=["Post"])
def get_image_class():
    if request.method == "POST":
        image = request.files["file"]

        # Save the model
        image.save(input_image_path)

        # Load the model
        img = tf.keras.utils.load_img(input_image_path, target_size=(img_height, img_width))


        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = ml_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        output = {"class": class_names[np.argmax(score)], "confidence(%)": 100 * np.max(score)}
        return output

if __name__ == '__main__':
    app.run(debug=True)