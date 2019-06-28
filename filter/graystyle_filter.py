from abstract.frame_filter import FrameFilter
import cv2


class GraystyleFilter(FrameFilter):

    def do_filter(self, original_frame, filtered_frame, output_frame):
        return cv2.cvtColor(filtered_frame, cv2.COLOR_RGB2GRAY), output_frame

