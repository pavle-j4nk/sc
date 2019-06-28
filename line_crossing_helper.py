import cv2


def get_number_rects(frame):
    img = 255 - frame
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    rects = []
    for c in contours:
        ca = cv2.contourArea(c)
        if ca < 100 or ca > 500:
            continue

        x, y, w, h = cv2.boundingRect(c)
        rects.append(((x, y), (x + w, y + h)))

    return rects


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None, None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


