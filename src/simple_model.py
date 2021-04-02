import numpy as np

class SimpleModel:
    def __init__(self, thresh=0.248):
        self.thresh = thresh
    def predict(self, hand):
        # compute distance
        p1_x = hand[0][12]
        p1_y = hand[0][13]
        p2_x = hand[0][60]
        p2_y = hand[0][61]
        d = np.square(p1_x - p2_x) + np.square(p1_y - p2_y)
        d = np.sqrt(d)
        if d > self.thresh:
            return [0]
        return [1]

