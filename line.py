import cv2
import numpy

from abstract.frame_filter import FrameFilter
from filter.number_areas_filter import NumberAreaFilter
from numberarea import NumberArea
from geometry import line_length


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
        self.len = line_length((self.x1, self.y1), (self.x2, self.y2))

        cv2.line(output_frame, (self.x1, self.y1), (self.x2, self.y2), (255, 255, 255), 2)
        return filtered_frame, output_frame

    def point_line_side(self, point, lp1, lp2):
        # d=(x−x1)(y2−y1)−(y−y1)(x2−x1)
        return (point[0] - lp1[0]) * (lp2[1] - lp1[1]) - (point[1] - lp1[1]) * (lp2[0] - lp1[0])

    def point_line_distance(self, point, lp1, lp2):
        return numpy.linalg.norm(numpy.cross(lp2 - lp1, lp1 - point)) / numpy.linalg.norm(lp2 - lp1)

    def rect_center(self, rect):
        return rect.p1[0] + (abs(rect.p1[0] - rect.p2[0])) / 2, rect.p1[1] + (abs(rect.p1[1] - rect.p2[1])) / 2

    def is_point_in_rect(self, point, rect, e=0):
        # return (min(rect.p1[0], rect.p2[0]) - e < point[0] < max(rect.p1[0], rect.p2[0]) + e) and (
        #         min(rect.p1[1], rect.p2[1]) - e < point[1] < max(rect.p1[1], rect.p2[1]) + e)
        return (min(rect[0][0], rect[1][0]) - e < point[0] < max(rect[0][0], rect[1][0]) + e) and (
                min(rect[0][1], rect[1][1]) - e < point[1] < max(rect[0][1], rect[1][1]) + e)

    def is_rect_on_line(self, rect):
        dx = (rect.p2[0] - rect.p1[0]) / 2
        dy = (rect.p2[1] - rect.p1[1]) / 2

        p1 = (rect.p1[0] + dx, rect.p1[1])
        p2 = (rect.p2[0] - dx, rect.p2[1])
        p3 = (rect.p1[0], rect.p1[1] + dy)
        p4 = (rect.p2[0], rect.p2[1] - dy)

        tolerance = 15

        if not (self.is_point_in_rect(p1, self.line_area, tolerance) and
                self.is_point_in_rect(p2, self.line_area, tolerance) and
                self.is_point_in_rect(p3, self.line_area, tolerance) and
                self.is_point_in_rect(p4, self.line_area, tolerance)):
            return False

        min_d = numpy.sqrt((rect.p1[0] - rect.p2[0]) ** 2 + (rect.p1[1] - rect.p2[1]) ** 2)
        rect_center = self.rect_center(rect)
        d = self.point_line_distance(rect_center, numpy.array((self.x1, self.y1)), numpy.array((self.x2, self.y2)))

        return d < min_d

    def line_intersection(self, line1, line2, e):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        if not ((min(line1[0][0], line1[1][0]) - e <= x <= e + max(line1[0][0], line1[1][0])) and (
                min(line1[0][1], line1[1][1]) - e <= y <= e + max(line1[0][1], line1[1][1]))):
            return None
        if not ((min(line2[0][0], line2[1][0]) - e <= x <= e + max(line2[0][0], line2[1][0])) and (
                min(line2[0][1], line2[1][1]) - e <= y <= e + max(line2[0][1], line2[1][1]))):
            return None

        return x, y

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
