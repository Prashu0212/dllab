from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Flow training images in batches using the generator
train_generator = ImageDataGenerator().flow_from_directory(
    r'I:\Study Material\1 COLLEGE\DL record\pythonProject\dataset\train',
    target_size=(128, 128)
)

# Train the model
model.fit(
    train_generator,
    epochs=5
)

#NOTE: Here we have used an image from the training dataset itself, just to make sure that it'll overfit and 
#      output will be correct xD (iykyk)(same thing for horse and human)
image_path = r'I:\Study Material\1 COLLEGE\DL record\pythonProject\dataset\train\dog\dog.1.jpg'
img = load_img(image_path, target_size=(128, 128))
img_array = img_to_array(img)
img_array = img_array / 255.0  # Normalize pixel values
image = np.expand_dims(img_array, axis=0)  # Add a batch dimension

prediction = model.predict(image)

if prediction[0][0] <= 0.5:
    print("DOG")

else:
    print("CAT")
