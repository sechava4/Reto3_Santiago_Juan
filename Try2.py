import cv2
import numpy as np


img = cv2.imread('RETO.JPG')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
kernel = np.ones((5,5),np.uint8)

lower_green = np.array([50,70,50])
upper_green = np.array([100,255,255])

mask = cv2.inRange(hsv, lower_green, upper_green)

closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


res = cv2.bitwise_and(img,img, mask= mask)
cv2.imshow("img",img)
cv2.imshow("mask",closing)
cv2.imshow("res",res)
cv2.waitKey(0)