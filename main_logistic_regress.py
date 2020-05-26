import visualizer
import numpy as np
from cv2 import imread
from os import system
from time import sleep
from pydirectinput import keyUp, keyDown
from gesture_ML_logistic_regress import sigmoid


def check_frame(w_parameter, b_parameter):
    check_img = np.array(imread("livefeed.png"))
    if check_img is not None:
        check_img.resize((check_img.shape[0] * check_img.shape[1] * check_img.shape[2], 1))
        check_img = check_img / 255
        y = sigmoid(np.dot(w_parameter.T, check_img) + b_parameter)
        return np.floor(2 * y)


if __name__ == '__main__':
    paras = np.load("lregression_parameters.npy", allow_pickle=True)
    w = paras[0]
    b = paras[1]
    del paras
    G = visualizer.GestureDetector()
    started, start_key = False, None

    while G.showing:
        G.show_video_feed()

        started = True
        if G.out is not None:
            response = check_frame(w, b)
            if response == 1:
                print("wot")
            elif response == 0:
                keyDown('shiftleft')
                sleep(0.005)
                keyUp('shiftleft')
                print("hand")
