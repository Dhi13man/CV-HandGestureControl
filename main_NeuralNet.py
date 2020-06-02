import visualizer
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from time import sleep
from joblib import load
from pydirectinput import keyUp, keyDown


class Models:
    def __init__(self, model_file):
        self.cnn = load_model(model_file)

    def check_frame(self):
        check_img = load_img("livefeed.png")
        if check_img is not None:
            check_img = img_to_array(check_img)
            check_img = np.expand_dims(check_img, axis=0)
            result = self.cnn.predict(check_img)[0]
            temp = max(result)
            for i, this in enumerate(result):
                if temp == this:
                    return i


if __name__ == '__main__':
    G = visualizer.GestureDetector()
    started, start_key = False, None
    model_name = 'CNN_DL.h5'
    Network = Models(model_name)

    while G.showing:
        G.show_video_feed(50, True)
        started = True
        if G.out is not None:
            response = Network.check_frame()
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
