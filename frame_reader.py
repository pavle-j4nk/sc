import cv2

from abstract.frame_display import FrameDisplay
from abstract.frame_filter import FrameFilter


class VideoReader:
    def __init__(self, file, fd: FrameDisplay = None):
        self.file = file
        self.fd = fd
        self.filters = []

    def add_filter(self, fp: FrameFilter):
        self.filters.append(fp)

    def set_frame_display(self, fd: FrameDisplay):
        self.fd = fd

    def read_all_frames(self):
        vcap = cv2.VideoCapture(self.file)
        fps = vcap.get(cv2.CAP_PROP_FPS)

        if self.fd:
            self.fd.set_delay(int(1000 / fps))

        while vcap.isOpened():
            r, frame = vcap.read()
            if not r:
                break

            filtered = frame.copy()
            output = frame.copy()

            for f in self.filters:
                filtered, output = f.do_filter(frame, filtered, output)

            if self.fd:
                self.fd.show(output)
