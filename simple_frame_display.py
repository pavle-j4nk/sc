from abstract.frame_display import FrameDisplay
import cv2
from time import time


class SimpleFrameDisplay(FrameDisplay):
    name = None
    pause = None

    lastFrameTime = None

    def __init__(self, name='Output', pause=False):
        self.name = name
        self.pause = pause
        self.lastFrameTime = time()

    def show(self, frame: []):
        cv2.imshow(self.name, frame)

        if self.pause:
            if cv2.waitKey() == ord("q"):
                exit(0)
        else:
            wait_time = self.delay - (time() - self.lastFrameTime)
            if wait_time > 0 and cv2.waitKey(1) == ord("q"):
                exit(0)

            self.lastFrameTime = time()
