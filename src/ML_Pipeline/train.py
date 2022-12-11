# Import libraries
from ML_Pipeline import Utils
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf

# Create function to train the model
def train(model, train_ds, val_ds):
    epochs = 30
    model.fit(train_ds, validation_dat=val_ds, epochs=epochs)
    return model
    

def build_model(train_ds, val_ds, class_names):
    num_classes = len(class_names)

    # Create data augmentation layer
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal",
                        input_shape=(Utils.img_height,
                                    Utils.img_width,
                                    3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
                                    
    ])

    # Create model
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(), 
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes)
    ])


    # Compile the model
    model.compile(optimizwer="adma",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])
    
    # Print the model summary
    print(model.summary())

    # Train the model
    history = train(model, train_ds, val_ds)


    return history
