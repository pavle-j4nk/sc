from keras.models import model_from_json
import numpy as np

class NumberRecognizer:

    def __init__(self):
        nn_json_file = open('model.json', 'r')
        nn_json = nn_json_file.read()
        nn_json_file.close()
        self.nn = model_from_json(nn_json)
        self.nn.load_weights("model.h5")

    def recognize(self, imgs):
        # imgs = self.calc_hog_features(imgs)
        red, kolona = imgs.shape[1:]
        imgs = imgs.reshape(imgs.shape[0], red, kolona, 1)

        predictions = self.nn.predict(imgs)
        results = self.get_results(predictions)
        return results

    def get_results(self, predictions):
        results = []
        for p in predictions:
            results.append(self.get_index_of_max(p))

        return results

    def get_index_of_max(self, arr):
        arr[arr < 0.5] = 0
        return np.unravel_index(np.argmax(arr, axis=None), arr.shape)[0]
