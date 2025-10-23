import tensorflow as tf
from tensorflow.keras import layers, models

print("TensorFlow version:", tf.__version__)
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
# Normalize images
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
# Resize images to 224x224 
resize_layer = tf.keras.Sequential([
    layers.Resizing(224, 224)
])

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).map(lambda x, y: (resize_layer(x), y))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32).map(lambda x, y: (resize_layer(x), y))

base_resnet = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_resnet.trainable = False  # freeze backbone

resnet_model = models.Sequential([
    base_resnet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")  # CIFAR-10 has 10 classes
])

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training ResNet50...")
resnet_model.fit(train_ds, validation_data=val_ds, epochs=3)

base_resnet.trainable = True
for layer in base_resnet.layers[:-30]:
    layer.trainable = False

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # lower LR
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Fine-tuning ResNet50...")
resnet_model.fit(train_ds, validation_data=val_ds, epochs=2)
