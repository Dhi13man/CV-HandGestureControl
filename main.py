import visualizer
import numpy as np
from cv2 import imread
from time import sleep
from joblib import load
from pydirectinput import keyUp, keyDown
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def get_mode(of_this_list):
    prediction = Counter(of_this_list)
    return prediction.most_common(n=1)[0][0]


class Models:
    def __init__(self, model_file):
        self.model = load(model_file)

    def check_frame(self):
        check_img = np.asarray(imread("livefeed.png"))
        if check_img is not None:
            check_img.resize((1, check_img.shape[0] * check_img.shape[1] * check_img.shape[2]))
            check_img = check_img / 255
            check_img = np.uint8(check_img)
            return self.model.predict(check_img)[0]


if __name__ == '__main__':
    G = visualizer.GestureDetector()
    started, start_key = False, None
    model_files = ['knn.pkl', 'svm_lin.pkl', 'rf.pkl']

    # Normal Loading of Models
    # knn = Models(model_files[0])
    # svm = Models(model_files[1])
    # rf = Models(model_files[2])

    # Multi-threaded loading of Models
    t_exec = ThreadPoolExecutor()
    init_threads = t_exec.map(Models, model_files)
    t2_exec = ThreadPoolExecutor()
    models = [result for result in init_threads]

    while G.showing:
        G.show_video_feed(50, True)
        started = True
        if G.out is not None:
            # Multi-threading each model's prediction procedure
            process_thread1 = t2_exec.submit(models[0].check_frame)
            process_thread2 = t2_exec.submit(models[1].check_frame)
            process_thread3 = t2_exec.submit(models[2].check_frame)
            predict_RF = process_thread3.result()
            predict_SVM = process_thread2.result()
            predict_KNN = process_thread1.result()

            response = get_mode([predict_RF, predict_SVM, predict_KNN])
            # 0: No Hand, 1: High Five, 2: Middle Finger, 3: V Sign
            print(response)
            
            # Code to perform actions based on gesture, below (uncomment and/or modify to use):
            # if response == 0:
            #     pass
            # elif response == 1:
            #     keyDown('w')
            #     sleep(0.15)
            #     keyUp('w')
            # elif response == 2:
            #     keyDown('shiftleft')
            #     sleep(0.005)
            #     keyUp('shiftleft')
            # elif response == 3:
            #     keyDown('shiftright')
            #     sleep(0.005)
            #     keyUp('shiftright')
            #     keyDown('w')
            #     sleep(0.02)
            #     keyUp('w')
