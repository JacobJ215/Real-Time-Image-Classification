# Import libraries
import pathlib
import subprocess

import tensorflow as tf

from ML_Pipeline import train
from ML_Pipeline import Utils
from ML_Pipeline.Preprocess import create_dataset
from ML_Pipeline.Utils import load_model, save_model

# Prompt user 
user_input = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value: "))

# Train
if user_input == 0:
    data_dir = pathlib.Path("../input/Training_data/")
    image_count = len(list(data_dir.glob('*/*')))
    print(f"Number of images for training: {image_count}")

    train_ds, val_ds, class_names = create_dataset(data_dir)
    ml_model = train.build_model(train_ds, val_ds, class_names)
    model_path = save_model(ml_model)
    print("The model was saved in", "../output/cnn-model")


# Predict
elif user_input == 1:
    model_path = "../output/cnn-model.h5"
    ml_model = load_model(model_path)

    test_data_dir = pathlib.Path("../input/Testing_Data/")
    image_count = len(list(test_data_dir.glob('*/*')))
    print(f"Number of images for testing {image_count}")

    # Create dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir, 
        seed=42,
        image_size=(Utils.img_height, Utils.img_width),
        batch_size=Utils.batch_size
    )

    # Make prediction
    prediction = ml_model.predict(test_ds)

    print(prediction)
    print(ml_model.evaluate(test_ds))

# Deploy
else:
    # For production deployment
    '''process = subprocess.Popen(['sh', 'ML_Pipeline/wsgi.sh'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )'''

    # For development deployment
    process = subprocess.Popen(['python', 'src/ML_Pipeline/deploy.py'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True
                                )
    for stdout_line in process.stdout:
        print(stdout_line)

    stdout, stderr = process.communicate()
    print(stdout, stderr)

