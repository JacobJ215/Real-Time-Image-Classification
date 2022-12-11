# Import libraries
import tensorflow as tf

batch_size = 32
img_height = 180
img_width = 180


# Function to cace data for tensorflow
def cache_data(train_ds, val_ds):
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


# Function to call dependent functions
def create_dataset(data_dir):
    print("Preprocessing has begun...")

    # Create training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_height,img_width),
        batch_size=batch_size
    )

    # Create validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, 
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_height, img_height),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(f"Class Names: {class_names}")
    print("Data loading has completed...")

    train_ds, val_ds = cache_data(train_ds, val_ds)

    print("Preprocessing is complete...")
    return train_ds, val_ds, class_names