from keras.utils import image_dataset_from_directory
from keras import Sequential
import keras.layers as layers
import matplotlib.pyplot as plt
import keras.losses as losses
import tensorflow as tf
from datetime import datetime

(train_ds, validation_ds) = image_dataset_from_directory(
    "train",
    labels="inferred",
    seed=0,
    label_mode="int",
    class_names=None,
    color_mode="grayscale",
    batch_size=64,
    image_size=(32, 32),
    shuffle=True,
    validation_split=0.3,
    subset="both",
    interpolation="gaussian",
)

class_names = train_ds.class_names

# this doesn't work:
# normalization_layer = Rescaling(1.0 / 255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")

# plt.show()

model = Sequential(
    [
        # layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(2),
    ]
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)


model.compile(
    optimizer="adam",
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(train_ds, validation_data=validation_ds, epochs=20)

plt.plot(history.history["val_accuracy"])
plt.show()

# save if you want
model.save(f"model_{datetime.now().strftime('%y-%m-%d_%H:%M')}")

# for neural network 128
# Epoch 10/10
# 547/547 [==============================] - 25s 45ms/step - loss: 0.4830 - accuracy: 0.7647 - val_loss: 0.5658 - val_accuracy: 0.7260

# for network 128->64->32
# Epoch 10/10
# 547/547 [==============================] - 25s 46ms/step - loss: 0.3860 - accuracy: 0.8223 - val_loss: 0.5849 - val_accuracy: 0.7448

# for network 128->64
# Epoch 10/10
# 547/547 [==============================] - 18s 32ms/step - loss: 0.3996 - accuracy: 0.8144 - val_loss: 0.5422 - val_accuracy: 0.7475

# for network 128->64, validation split 0.2 score is a bit worse

# for batch size 16, took longer, the learning rate was more stable, the difference between test and val is smaller, but the val acc is pretty much the same
# Epoch 10/10
# 1094/1094 [==============================] - 33s 30ms/step - loss: 0.4347 - accuracy: 0.7955 - val_loss: 0.5540 - val_accuracy: 0.7433

# for batch size 64
# Epoch 10/10
# 274/274 [==============================] - 27s 99ms/step - loss: 0.3840 - accuracy: 0.8257 - val_loss: 0.6302 - val_accuracy: 0.7175

# for epoch 20, model is overlearned, too much test accuracy, the val acc is going down,
# Epoch 20/20
# 547/547 [==============================] - 39s 70ms/step - loss: 0.2529 - accuracy: 0.8917 - val_loss: 1.1396 - val_accuracy: 0.6752

# for 20 epoch, 16 batch, 128->64->32, the results in 11th epoch weren't enough
