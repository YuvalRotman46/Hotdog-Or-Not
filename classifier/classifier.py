"""
This is a file which contains the classifying code.
The classifier here can classify images to HotDogs and UnHotDogs.
Inspired by "Silicone Valley".
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import classifier.data_preprocessor

EPOCHS = 80
BATCH_SIZE = 50

IMG_DIMS = classifier.data_preprocessor.IMG_DIMS
MODEL_PATH = r'model.h5'
OUTPUT_MAP = ["Hot Dog", "not a Hot Dog"]


class Core:
    def __init__(self):
        self.model = None

    def train(self):
        model = Core.create_model()
        Core.build_model(model)
        Core.train_model(model)
        self.model = model

    def load(self):
        self.model = Core.load_model()

    def classify(self, img_relative_path):
        img_vector = classifier.data_preprocessor.get_image_vector(img_relative_path, IMG_DIMS)
        output = self.model.predict(np.array([img_vector]))
        output_index = np.argmax(output)
        return OUTPUT_MAP[output_index]

    def __str__(self):
        return "{HotDog Classifier}"

    @staticmethod
    def create_model():
        model = keras.models.Sequential()
        # input + first layers
        model.add(keras.layers.Dense(15, input_shape=[IMG_DIMS[0]**2*3]))
        # hidden layers
        model.add(keras.layers.Dense(10))
        model.add(keras.layers.Dense(10))
        # output layer
        model.add(keras.layers.Dense(2, activation='softmax'))
        return model

    @staticmethod
    def build_model(model):
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    @staticmethod
    def train_model(model):
        train_set, test_set = classifier.data_preprocessor.load_dataset()
        train_x = train_set[0]
        train_y = train_set[1]

        test_x = test_set[0]
        test_y = test_set[1]

        history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS,
                            validation_data=(test_x, test_y), verbose=2)
        model.save(MODEL_PATH)

    @staticmethod
    def load_model():
        model = keras.models.load_model(MODEL_PATH)
        return model

