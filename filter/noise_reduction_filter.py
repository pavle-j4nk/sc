from abstract.frame_filter import FrameFilter
import cv2


class NoiseReductionFilter(FrameFilter):
    """
    Ukljanja krstice
    """

    def do_filter(self, original_frame, filtered_frame, output_frame):
        self.remove_green(filtered_frame)

        # filtered_frame = self.dilate(filtered_frame)

        return filtered_frame, output_frame

    def remove_green(self, frame):
        mask = (frame[:, :, 1] > 127) & (frame[:, :, 0] < 127) & (frame[:, :, 2] < 127)
        frame[mask] = 0

    def dilate(self, frame):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.dilate(frame, kernel, iterations=1)
