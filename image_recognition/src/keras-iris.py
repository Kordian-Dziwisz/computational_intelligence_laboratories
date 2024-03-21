import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from keras import losses
from keras import activations

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler()
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# skaluje zestaw danych, standardowy wzór to z = (x - u) / s, gdzie:
# u -> średnia
# s -> odchylenie standardowe
X_scaled = scaler.fit_transform(X)
# tutaj aktywuje się StandardScaler

# Encode the labels
encoder = OneHotEncoder(sparse_output=False)
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# https://en.wikipedia.org/wiki/One-hot
# one-hot enconding jest to metoda zapisywania danych, w której istnieje tylko jeden wyznacznik stanu wysokiego (1) a reszta jest stanem niskim (0). Podobne do systemu unarnego
# po prostu zapiszujesz wartość, np 8, na 8 bitach, i każda wartość to inny aktywny bit
y_reshaped = y.reshape(-1, 1)
# reshape zmienia kształt danych w tabeli, w tym wypadku z listy o długości 150 na 150 list jednoelementowych (-1 oznacza, że dzieli na tyle tabel rozmiaru następnego argumentu ile może)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation=activations.relu, input_shape=(X_train.shape[1],)),
    Dense(64, activation=activations.relu),
    Dense(y_encoded.shape[1], activation='softmax')
])
# https://www.w3schools.com/python/numpy/numpy_array_shape.asp
# X_train.shape i Y_train shape określają rozmiary tabeli w numpy, w tym wypadku 4 dane wejsciowe i 3 wyjściowe
# relu - ~ 1 / ~ 0.9
# sigmoid - ~ 0.95  / ~ 0.8
# softmax jest beznadziejne
# tanh - ~ 1 / ~ 0.95
# tanh jest potencjalnie lepsze

# Compile the model
model.compile(optimizer='sgd', loss=losses.binary_crossentropy, metrics=['accuracy'])
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
# optimizer adadelta jest beznadziejny
# adamax jest naprawdę szybko i skutecznie sprawdza walidacyjny, nie radzi sobie tak dobrze z treningowym
# ftrl jest beznadziejny
# sgd (gradient descent with momentum) jest dobry, ale gorszy niż adam
# https://www.tensorflow.org/api_docs/python/tf/keras/losses
# binary crossentropy - dobry, ale gorszy
# categorical hinge - szalone rezultaty najpierw dobra celność na walidacyjnym, potem spada, a potem rośnie. Generalnie gorszy od categorical crossentropy


# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('iris_model.h5')

# Plot and save the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


