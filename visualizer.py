import cv2
from imutils import resize


class GestureDetector:
    def __init__(self, para=None):
        self.check_paras = para
        self.bg, self.area, self.frame_copy = None, None, None
        self.key = None
        self.showing = True
        self.num_frames = 0
        self.camera = cv2.VideoCapture(0)
        self.out = None

    def end_feed(self):
        cv2.destroyAllWindows()
        self.camera.release()

    def loop_feed(self):
        while self.showing:
            self.show_video_feed()

    def avg_backGr_get(self, image, weight):
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        cv2.accumulateWeighted(image, self.bg, weight)

    def isolate_hand(self, image, threshold=25):
        difference = cv2.absdiff(self.bg.astype("uint8"), image)
        threshold_area = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)[1]
        (contours, _) = cv2.findContours(threshold_area.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        else:
            isolated_hand = max(contours, key=cv2.contourArea)
            return threshold_area, isolated_hand

    def show_video_feed(self, calibration_no=50, show_windows=True):
        weight = 0.5
        # Keep Width 240, Height 225
        # Camera POV, Left Edge: 450, 690 ; Right Edge:
        sc_right, sc_left, sc_top, sc_bottom = 450, 690, 70, 295
        grabbed, frame = self.camera.read()
        frame = resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        self.frame_copy = frame.copy()

        # TODO Try modifying
        region = frame[sc_top:sc_bottom, sc_right:sc_left]
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        region = cv2.GaussianBlur(region, (7, 7), 0)

        if self.num_frames < calibration_no:
            self.avg_backGr_get(region, weight)
            temp = cv2.imread("none.png")
            cv2.imwrite("livefeed.png", temp)
            del temp
        elif self.num_frames == calibration_no:
            print("CALIBRATION COMPLETE!!!!")
        else:
            hand = self.isolate_hand(region)
            if hand is not None:
                self.area, hand_cont = hand
                cv2.drawContours(self.frame_copy, [hand_cont + (sc_right, sc_top)], -1, (0, 0, 255))
                if show_windows:
                    cv2.imshow('Isolated hand', self.area)

        cv2.rectangle(self.frame_copy, (sc_left, sc_top), (sc_right, sc_bottom), (0, 255, 0), 2)
        self.num_frames += 1
        if show_windows:
            cv2.imshow("Overall image", self.frame_copy)
        self.save_isolated_frame()

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            self.showing = False
            self.end_feed()
            return
        elif keypress == ord("r"):
            print("RECALIBRATE")
            self.camera.release()
            self.bg, self.area = None, None
            self.num_frames = 0
            self.camera = cv2.VideoCapture(0)
            self.show_video_feed()
            return
        else:
            self.key = keypress
            self.showing = True

    def save_isolated_frame(self):
        if self.area is not None:
            self.out = self.area
            cv2.imwrite("livefeed.png", self.area)


if __name__ == '__main__':
    G = GestureDetector()
    G.loop_feed()
