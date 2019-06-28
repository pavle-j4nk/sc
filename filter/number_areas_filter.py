from operator import attrgetter

import cv2
import numpy

from abstract.frame_filter import FrameFilter
from number_recognizer import NumberRecognizer
from numberarea import NumberArea
import numpy as np


class NumberAreaFilter(FrameFilter):
    rects = []
    count = 0

    def __init__(self, nr: NumberRecognizer):
        self.nr = nr

    def do_filter(self, original_frame, filtered_frame, output_frame):
        self.recognize_numbers(filtered_frame)
        self.update_rects_age()
        self.update_rects(filtered_frame)
        self.remove_old_rects()
        self.draw_rects(output_frame)

        return filtered_frame, output_frame

    def recognize_numbers(self, frame):
        for r in self.rects:
            if r.age > 20 and r.time_since_update == 0:
                num_only = self.prepare_for_recognition(r, frame)
                result = self.nr.recognize(numpy.array([num_only]))
                r.add_value(int(result[0]))

    def prepare_for_recognition(self, rect: NumberArea, filtered):
        x = rect.p1[0]
        y = rect.p1[1]
        h = rect.p2[1] - rect.p1[1]
        w = rect.p2[0] - rect.p1[0]

        num_img = filtered[y:y + h, x:x + w]
        num_img = self.deskew(num_img)

        # cv2.imshow("pre-" + str(rect.id), num_img)
        # cv2.waitKey()

        _, num_img_inv = cv2.threshold(num_img, 100, 255, cv2.THRESH_BINARY_INV)
        cnts, hierarchy = cv2.findContours(num_img_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        biggest = self.biggest_cnt(cnts)
        x, y, w, h = cv2.boundingRect(biggest)
        content_olny = num_img[y:y + h, x:x + w]
        # cv2.imshow("thresholded-" + str(rect.id), content_olny)
        # cv2.waitKey()

        centered = numpy.zeros((28, 28))
        centered_x = int(28 / 2 - w / 2)
        centered_y = int(28 / 2 - h / 2)
        centered[centered_y: centered_y + h, centered_x: centered_x + w] = content_olny

        # cv2.imshow(str(rect.id), centered)
        # cv2.waitKey()

        return centered

    def biggest_cnt(self, cnts):
        biggest_cnt = None
        biggest_area = -1
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 27*27 and area > biggest_area:
                biggest_area = area
                biggest_cnt = c
        return biggest_cnt

    def deskew(self, img):
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            # no deskewing needed.
            return img.copy()
        # Calculate skew based on central momemts.
        skew = m['mu11'] / m['mu02']
        # Calculate affine transform to correct skewness.
        SZ = 28
        M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
        # Apply affine transform
        img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img

    def update_rects_age(self):
        for r in self.rects:
            r.time_since_update += 1
            r.age += 1

    def update_rects(self, img):
        cnts = self.get_number_contours(img)
        for c in cnts:
            ca = cv2.contourArea(c)
            if ca < 80 or ca > 500:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if not (w > 20 or h > 20):
                continue

            r = ((x, y), (x + w, y + h))

            cr = self.find_original_rect(r)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = numpy.int0(box)

            if cr is not None:
                # update existing
                cr.p1 = r[0]
                cr.p2 = r[1]
                cr.path.append((r[0], r[1]))
                cr.contours = box
                cr.time_since_update = 0
            else:
                # create new
                rect = NumberArea(r[0], r[1])
                rect.id = self.count
                self.count += 1
                self.rects.append(rect)
                rect.contours = box

    def remove_old_rects(self):
        for i in reversed(range(len(self.rects))):
            if self.rects[i].time_since_update > 30:
                del self.rects[i]
                continue

    def draw_rects(self, img):
        for r in self.rects:
            cv2.rectangle(img, r.p1, r.p2, r.color, 1)
            cv2.putText(img, str(r.id) + "|" + str(r.get_value()) + "|" + str(r.crossings), r.p1,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color=(255, 255, 255),
                        lineType=cv2.LINE_AA)
            # cv2.drawContours(img, [r.contours], 0, (0, 0, 255), 1)

    def get_number_contours(self, frame):
        frame = frame.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        frame = cv2.dilate(frame, kernel, iterations=1)
        img = 255 - frame
        ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        return contours

    def find_original_rect(self, rect):
        or_rects = []

        for or_r in self.rects:
            pos_tolerance = 10
            size_tolerance = 10

            # speed_x = (or_r.p1[0] - or_r.start[0]) / (or_r.age + 1)
            # speed_y = (or_r.p1[1] - or_r.start[1]) / (or_r.age + 1)
            aprox_x = or_r.p1[0] + int((or_r.time_since_update + 1))
            aprox_y = or_r.p1[1] + int((or_r.time_since_update + 1))

            h_old, w_old = self.rect_dims(or_r)
            h_new, w_new = self.rect_dims(rect)

            if abs(aprox_x - rect[0][0]) < pos_tolerance and abs(aprox_y - rect[0][1]) < pos_tolerance and abs(
                    h_old - h_new) < size_tolerance and abs(w_old - w_new) < size_tolerance:
                or_rects.append(or_r)

        if len(or_rects) > 0:
            return max(or_rects, key=attrgetter('age'))
        else:
            return None

    def longest_contour_line(self, cnts):
        lines = []
        for i in range(-1, len(cnts) - 1):
            lines.append([cnts[i], cnts[i + 1]])

        longest_line = lines[0]
        longest_val = -1

        for l in lines:
            length = line_length(l[0], l[1])
            print(length)
            if length > longest_val:
                longest_line = l
                longest_val = length

        return longest_line

    def line_length(self, w, v=(0, 0)):
        return sqrt((v[0] + w[0]) ** 2 + (v[1] + w[1]) ** 2)

    def rect_dims(self, rect):
        if isinstance(rect, NumberArea):
            h = rect.p2[1] - rect.p1[1]
            w = rect.p2[0] - rect.p1[0]
        else:
            h = rect[1][1] - rect[0][1]
            w = rect[1][0] - rect[0][0]

        return h, w
