import numpy as np
from cv2 import imread
import os


def sigmoid(array):
    return 1 * (1/(1 + np.exp(-array)))


class Detector:
    def __init__(self, no_of_iterations, learning_rate):
        self.location, self.c = None, 0
        self.w, self.b = np.array([]), 0.0
        self.costs = []
        self.x, self.y = np.array([[]]), np.array([[]])
        self.load_dataset("hi5")
        self.load_dataset("midfin")

        self.process_dataset()

        self.init_parameters()
        self.optimize_grad_desc(no_of_iterations, learning_rate)
        np.save("parameters", [self.w, self.b])
        print("Parameters Saved!")

    def load_dataset(self, location):
        self.location = "Datasets\\" + location + '\\'
        for file in os.listdir(self.location):
            image = np.array(imread(self.location + file))
            image.resize((image.shape[0]*image.shape[1]*image.shape[2], 1))
            if self.x.size == 0:
                self.x = np.hstack([image])
                self.y = [self.c]
            else:
                self.x = np.hstack([self.x, image])
                self.y = np.hstack([self.y, self.c])
        self.c += 1

    def process_dataset(self):
        self.x = self.x/255

    def init_parameters(self):
        self.w = np.zeros((self.x.shape[0], 1))
        self.b = 0.0

    def propagate(self):
        n = self.x.shape[1]
        A = sigmoid(np.dot(self.w.T, self.x) + self.b)
        cost = -np.sum(self.y * np.log(A) + (1 - self.y) * np.log(1 - A))/n

        dw = np.dot(self.x, (A - self.y).T)/n
        db = np.sum(A - self.y)/n

        cost = np.squeeze(cost)
        grads = {"dw": dw, "db": db}
        return grads, cost
    
    def optimize_grad_desc(self, num_iter, l_rate):
        for i in range(num_iter):
            print(i)
            grads, cost = self.propagate()
            dw, db = grads["dw"], grads["db"]

            self.w -= l_rate*dw
            self.b -= l_rate*db

            if i % 100 == 0:
                self.costs.append(cost)


if __name__ == '__main__':
    x = Detector(no_of_iterations=2000, learning_rate=0.005)
