from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

print('Train: X = ', trainX.shape)
print('Test: X = ', testX.shape)

for i in range(1, 10):
    plt.subplot(3, 3, i)

    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))

plt.show()

def model_arch():
	models = Sequential()

	models.add(Conv2D(64, (5, 5),
					padding="same",
					activation="relu",
					input_shape=(28, 28, 1)))

	models.add(MaxPooling2D(pool_size=(2, 2)))
	models.add(Conv2D(128, (5, 5), padding="same",
					activation="relu"))

	models.add(MaxPooling2D(pool_size=(2, 2)))
	models.add(Conv2D(256, (5, 5), padding="same",
					activation="relu"))

	models.add(MaxPooling2D(pool_size=(2, 2)))

	models.add(Flatten())
	models.add(Dense(256, activation="relu"))
	models.add(Dense(10, activation="softmax"))

	return models


model = model_arch()

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.summary()

history = model.fit(
    trainX.astype(np.float32), trainy.astype(np.float32),
    epochs=10,
    steps_per_epoch=100,
    validation_split=0.33
)

labels = ['t_shirt', 'trouser', 'pullover',
		'dress', 'coat', 'sandal', 'shirt',
		'sneaker', 'bag', 'ankle_boots']


print(len(testX))
predictions = model.predict(testX[6:7])
label = labels[np.argmax(predictions)]

print(label)
plt.imshow(testX[6:7][0])
plt.show()

