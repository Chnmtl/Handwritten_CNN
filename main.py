import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.utils import to_categorical

# Load the data and split it into train set and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

# Get the image shape
print('x_train image shape: ', x_train.shape)
print('x_test image shape: ', x_test.shape)

# Take a look at the first image in the training data set
print('First image: /n ', x_train[0])

# Print the image label
print('Label of the first image: ', y_train[0])

# Show image as a picture
# print('First images picture: ', plt.imshow(x_train[0]))

# Shorten the data so it will take less time to compile
x_train = x_train[:15000]
y_train = y_train[:15000]

x_test = x_test[:2500]
y_test = y_test[:2500]

print('Shortened x_train data: ', x_train.shape)
print('Shortened x_test data: ', x_test.shape)

# Reshape the data to fit the model
x_train = x_train.reshape(15000, 28, 28, 1)
x_test = x_test.reshape(2500, 28, 28, 1)

# One-Hot Encoding: Encode target data to convert labels into a set of thin numbers to implement to the NN
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Print the new label
print('New y label: ', y_train_one_hot[0])

# Build the CNN model
model = Sequential()

# add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))    # Convolution Layer
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())    # Flattening Layer
model.add(Dense(10, activation='softmax'))      # Fully-connected Layer

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot), epochs=3)

# Visualize the models accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Show predictions as probabilities for the first 4 images in the test set
predictions = model.predict(x_test[:4])
print('Predictions as probabilities: ', predictions)

# Print our predictions as number labels for the first 4 images
print('Predictions as number llabels: : ',  np.argmax(predictions, axis=1))

# Print the actual labels
print('Actual labels: ', y_test[:4])

