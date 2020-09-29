"""
This is a file which contains the classifying code.
The classifier here can classify images to HotDogs and UnHotDogs.
Inspired by "Silicone Valley".
"""


class Core:
    def __init__(self):
        raise NotImplementedError()

    def train_model(self):
        raise NotImplementedError()

    def load_model(self):
        raise NotImplementedError()

    def classify(self, img_relative_path):
        raise NotImplementedError()

    def __str__(self):
        return "{HotDog Classifier}"

    @staticmethod
    def create_model():
        raise NotImplementedError()


