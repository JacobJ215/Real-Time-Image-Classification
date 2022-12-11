# Import libraries
import keras

# Declare variables
batch_size = 32
img_height = 180
img_width = 180

# Create function to save model
def save_model(model):
    model.save("../output/cnn-model.h5")

    return True

# Create function to load model
def load_model(model_path):
    model = None
    try:
        model = keras.models.load_model(model_path)
    except:
        print("Please enter the correct path")
        exit(0)

    return model