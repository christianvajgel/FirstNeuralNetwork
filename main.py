from keras.datasets import mnist
from matplotlib import pyplot

# Data Charge
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Image plotting
for i in range(9):
    pyplot.subplot(3, 3, i + 1)
    pyplot.imshow(train_images[i], cmap=pyplot.get_cmap('gray'))

pyplot.show()

# Modelling neural network
from keras import models
from keras import layers

# Raw model of the network meanwhile it is sequential
network = models.Sequential()

# Add of the processing layers to the network

# First processing layer:
layer1 = layers.Dense(512, activation='relu', input_shape=(28 * 28, ))
network.add(layer1)

# Diagnostic layer
layer2 = layers.Dense(10, activation='softmax')
network.add(layer2)

# Compile network
# Involve: Choose the optimization function, choose the loss function, choose metrics
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_accuracy = network.evaluate(test_images, test_labels)