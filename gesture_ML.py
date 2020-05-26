import numpy as np
from cv2 import imread
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from joblib import dump
from concurrent.futures import ProcessPoolExecutor


class Detector:
    def __init__(self, total_training_number=100):
        self.location, self.c, self.accuracy = None, 0, 0
        self.costs = []
        self.x, self.y = np.array([[]]), None
        self.load_sets(total_training_number)

    def load_sets(self, load_total):
        def load_dataset(location, number=load_total):
            self.location = "Datasets\\" + location + '\\'
            c = 0
            for file in os.listdir(self.location):
                c += 1
                if c > number:
                    break
                image = np.array(imread(self.location + file))/255
                image = np.uint8(image)
                image.resize((1, image.shape[0]*image.shape[1]*image.shape[2]))
                if self.x.size == 0:
                    self.x = np.vstack([image])
                    self.y = [self.c]
                else:
                    self.x = np.vstack([self.x, image])
                    self.y.append(self.c)
                del image
            self.c += 1
        print("LOADING DATASET (each having n = %d)...\n\n" % load_total)
        load_dataset("0")
        load_dataset("hi5")
        load_dataset("midfin")
        load_dataset("v")

    def train_knn(self, no_of_neighbors=1):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
        x_train = np.uint8(x_train)
        x_test = np.uint8(x_test)
        print("TRAINING KNN MODEL...\n\n")
        classifier = KNeighborsClassifier(n_neighbors=no_of_neighbors, p=1)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        print('SAVING MODEL!!')
        dump(classifier, 'KNN.pkl', compress=True)
        print('Accuracy: ', self.accuracy)
        print("KNN Model Saved!\n")

    def train_random_forest(self, no_of_estimators=55):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
        x_train = np.uint8(x_train)
        x_test = np.uint8(x_test)
        print("TRAINING RANDOM FOREST MODEL...\n\n")
        classifier = RandomForestClassifier(criterion='entropy', n_estimators=no_of_estimators)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        print('SAVING MODEL!!')
        dump(classifier, 'rf.pkl')
        print('Accuracy: ', self.accuracy)
        print("Random Forest Model Saved!\n")

    def train_kernel_svm(self, kernel_type='linear'):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
        del self.x
        del self.y
        self.x, self.y = None, None
        x_train = np.uint8(x_train)
        x_test = np.uint8(x_test)
        print("TRAINING SVM MODEL...\n\n")
        classifier = SVC(kernel=kernel_type)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        print('SAVING MODEL!!')
        dump(classifier, 'svm_lin.pkl', compress=True)
        print('Accuracy: ', self.accuracy)
        print("SVM Model Saved!\n")


def get_model_accuracy(test, l_test=1, u_test=10, step_parameter_by=1, model='rf'):
    acc = []
    for parameter in range(l_test, u_test+1, step_parameter_by):
        if model == 'rf':
            test.train_random_forest(parameter)
        elif model == 'knn':
            test.train_knn(parameter)
        acc.append(test.accuracy)
    return acc


def train_knn():
    det = Detector(150)
    det.train_knn(2)


def train_rf():
    det = Detector(150)
    det.train_random_forest()


def train_svm():
    det = Detector(150)
    det.train_kernel_svm()


if __name__ == '__main__':
    with ProcessPoolExecutor() as ppe:
        t1 = ppe.submit(train_knn)
        t2 = ppe.submit(train_rf)
        t3 = ppe.submit(train_svm)

    while t1.running() or t2.running() or t3.running():
        print(". ", end='')
        pass

    print('All Parameters saved!')

