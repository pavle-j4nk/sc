from random import randint
import numpy as np

class NumberArea:

    start = None
    p1, p2 = None, None
    color = (255, 255, 255)

    age = 0
    crossings = 0

    speed = [0, 0]


    id = -1

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.start = p1
        self.color = (randint(0, 200), randint(0, 200), randint(0, 200))
        self.contours = None
        self.time_since_update = 0

        self.path = []
        self.values = []
        self.v = -1
        self.path.append((p1, p2))

        for v in range(0, 10):
            self.values.append(0)

    def add_value(self, value):
        # self.v = value
        self.values[value] += 1

    def get_value(self):
        return np.unravel_index(np.argmax(self.values, axis=None), np.array(self.values).shape)[0]
        # return self.v