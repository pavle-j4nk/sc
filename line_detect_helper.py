import cv2


def find_line(frame):
    # prepare
    img = frame.copy()
    isolate_line(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # determinalte line position
    bc = biggest_conture(contours)

    x, y, w, h = cv2.boundingRect(bc)
    return (x, y + h), (x + w, y)


def biggest_conture(contours):
    cnt = contours[0]
    largest_area = cv2.contourArea(cnt)
    for i in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            cnt = contours[i]
    return cnt


def isolate_line(frame):
    mask = (frame[:, :, 1] < 127) & (frame[:, :, 0] > 127) & (frame[:, :, 2] < 127)
    frame[mask.__invert__()] = [255, 255, 255]
    frame[mask] = 0
