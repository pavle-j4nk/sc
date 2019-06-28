from math import sqrt

import cv2
import numpy

from abstract.frame_filter import FrameFilter
from filter.number_areas_filter import NumberAreaFilter
from numberarea import NumberArea


class Line(FrameFilter):

    def __init__(self, naf: NumberAreaFilter, color_bottom_limits, on_cross=None):
        self.on_cross = on_cross
        self.color_bottom_limits = color_bottom_limits
        self.naf = naf
        self.x1 = self.y1 = self.x2 = self.y2 = -1
        self.len = 0

        self.line_area = None

        self.frameNumber = 0

        self.numbers_crossed = []

    def do_filter(self, original_frame, filtered_frame, output_frame):
        frame = original_frame.copy()
        self.frameNumber += 1

        if self.frameNumber > 10:
            for r in self.naf.rects:
                if self.point_line_distance(r.p1, numpy.array(self.line_area.p1), numpy.array(self.line_area.p2)) < 20\
                        and self.is_point_in_rect(r.p1, [self.line_area.p1, self.line_area.p2]):
                    if r not in self.numbers_crossed:
                        self.numbers_crossed.append(r)
                        r.crossings += 1

        if self.frameNumber > 50:
            cv2.line(output_frame, (self.x1, self.y1), (self.x2, self.y2), (255, 255, 255), 1)
            return filtered_frame, output_frame

        mask = ((frame[:, :, 0] < 255 - self.color_bottom_limits[0]) & (
                frame[:, :, 1] < 255 - self.color_bottom_limits[1]) & (
                        frame[:, :, 2] < 255 - self.color_bottom_limits[2]))
        frame[mask] = 0
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        frame = cv2.erode(frame, kernel, iterations=2)

        lines = cv2.HoughLinesP(frame, 1, numpy.pi / 200, 100, minLineLength=20, maxLineGap=20)

        x1s = y1s = x2s = y2s = 0

        for line in lines:
            for x1, y1, x2, y2 in line:
                x1s += x1
                x2s += x2
                y1s += y1
                y2s += y2
                # cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        x1 = int(x1s / len(lines))
        x2 = int(x2s / len(lines))
        y1 = int(y1s / len(lines))
        y2 = int(y2s / len(lines))

        if self.frameNumber <= 1:
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
        else:
            if x1 < self.x1:
                self.x1 = x1
                self.y1 = y1
            if x2 > self.x2:
                self.x2 = x2
                self.y2 = y2

        self.line_area = NumberArea((self.x1, self.y1), (self.x2, self.y2))
        self.len = self.line_length((self.x1, self.y1), (self.x2, self.y2))

        cv2.line(output_frame, (self.x1, self.y1), (self.x2, self.y2), (255, 255, 255), 2)
        return filtered_frame, output_frame

    def point_line_distance(self, point, lp1, lp2):
        return numpy.linalg.norm(numpy.cross(lp2 - lp1, lp1 - point)) / numpy.linalg.norm(lp2 - lp1)

    def is_point_in_rect(self, point, rect, e=0):
        return (min(rect[0][0], rect[1][0]) - e < point[0] < max(rect[0][0], rect[1][0]) + e) and (
                min(rect[0][1], rect[1][1]) - e < point[1] < max(rect[0][1], rect[1][1]) + e)

    def line_length(self, w, v=(0, 0)):
        return sqrt((v[0] + w[0]) ** 2 + (v[1] + w[1]) ** 2)

    def get_sum(self):
        sum = 0
        for n in self.numbers_crossed:
            sum += n.get_value()

        return sum

    @staticmethod
    def blue(naf: NumberAreaFilter):
        return Line(naf, [127, 0, 0])

    @staticmethod
    def green(naf: NumberAreaFilter):
        return Line(naf, [0, 127, 0])
