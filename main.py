
import cv2
import numpy as np


def redlight_detection():
    cap = cv2.VideoCapture("C:/Users/admin/Desktop/red light/v2.mov")
    video_cod = cv2.VideoWriter_fourcc(*'XVID')
    video_output = cv2.VideoWriter('captured_video.avi',video_cod,10,(int(cap.get(3)),int(cap.get(4))))

    while (1):
        _, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        mask=mask0+mask1

        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        video_output.write(frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    video_output.release()




if __name__ == '__main__':
    redlight_detection()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
