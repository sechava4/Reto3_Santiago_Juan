import cv2
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from matplotlib import pyplot as plt
from sklearn import linear_model
import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200)
print(ser.readline())

capture = cv2.VideoCapture(0)
first = True

while (True):

    _, frame = capture.read()
    cv2.imshow("f", frame)

    # img = cv2.imread('RETO.JPG')
    # frame = frame[0:1700, 0:1000]     #Se RECORTA IMAGEN
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    kernel = np.ones((4, 4), np.uint8)

    #desde la mesa
    #lower_green = np.array([52, 30, 110])
    #upper_green = np.array([68, 70, 140])

    lower_green = np.array([10, 60, 100])
    upper_green = np.array([99, 89, 150])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    closing = cv2.erode(closing, kernel, iterations=2)
    opening = cv2.dilate(closing, kernel, iterations=2)

    indices = np.where(opening == [255])
    # print (indices)
    coordinates = zip(indices[0], indices[1])
    # print (coordinates)
    X = indices[0]
    y = indices[1]

    y = np.transpose([y])
    X = np.transpose([X])

    ############RANSCAC###################

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor()
    if ((len(X)>=3) or (len(y)>=3)):

        ransac.fit(X, y)

        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        # Predict data of estimated models
        line_X = np.arange(X.min(), X.max())[:, np.newaxis]
        # line_y = lr.predict(line_X)
        line_y_ransac = ransac.predict(line_X)

        #############Line 1 Equation###############

        # Get 2 points from line 1

        points0 = [(line_X[0][0], line_y_ransac[0][0]), (line_X[1][0], line_y_ransac[1][0])]
        x_coords0, y_coords0 = zip(*points0)
        A0 = vstack([x_coords0, ones(len(x_coords0))]).T
        m0, c0 = lstsq(A0, y_coords0, rcond=None)[0]
        # print("Line Solution is y = {m}x + {c}".format(m=m0,c=c0))

        ################## Second RANSAC ###################################
        X2 = []
        y2 = []

        X[inlier_mask] = 0
        y[inlier_mask] = 0

        X2 = X[outlier_mask]
        y2 = y[outlier_mask]

        # Fit linear model with RANSAC algorithm

        ransac2 = linear_model.RANSACRegressor()
        if ((len(X2) >= 3) or (len(y2) >= 3)):
            ransac2.fit(X2, y2)

            inlier_mask2 = ransac2.inlier_mask_
            outlier_mask2 = np.logical_not(inlier_mask2)

            # Predict data of estimated models
            line_X2 = np.arange(X2.min(), X2.max())[:, np.newaxis]
            line_y2_ransac = ransac2.predict(line_X2)

            #############Line 2 Equation###############
            # Get 2 points from line 2

            points2 = [(line_X2[0][0], line_y2_ransac[0][0]), (line_X2[1][0], line_y2_ransac[1][0])]
            x_coords2, y_coords2 = zip(*points2)
            A2 = vstack([x_coords2, ones(len(x_coords2))]).T
            m2, c2 = lstsq(A2, y_coords2, rcond=None)[0]
            # print("Line2 Solution is y = {m}x + {c}".format(m=m2,c=c2))

            ########### Intersection ###########
            if(m2==0):
                print("No hay ")
            else:
                inter_x = ((c0 * m2) - (c2 * m0)) / (m2 - m0)
                inter_y = (inter_x - c2) / m2



                #####PLOT########

                lw = 2

                width, height, _ = frame.shape

                """
                plt.xlim(0, width)
                plt.ylim(0,height)
            
            
                plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
                         label='RANSAC regressor')
                plt.plot(line_X2, line_y2_ransac, color='blue', linewidth=lw,
                         label='RANSAC regressor2')
            
                plt.legend(loc='lower right')
                plt.xlabel("Input")
                plt.ylabel("Response")
                plt.show()
                """

                ######## Se calcula el error ########

                inter_x = int(round(inter_x))
                inter_y = int(round(inter_y))
                centro_y = int(round(width / 2))
                centro_x = int(round(height / 2))

                # Si el error es positivo - mover a la izq
                error = (round(inter_x - centro_x))

                if (first):
                    error_ant = error
                    first = False

                if (abs(error) > 100):
                    error = error_ant

                # Se envia a arduino via string
                msg = str(error)
                msg = msg + 'p'
                ser.write(msg.encode())

                # Se lee mensaje recibido del arduino
                #print(ser.readline())
                print(error)

                res = cv2.bitwise_and(frame, frame, mask=opening)
                cv2.circle(res, (centro_x, inter_y), 5, (0, 0, 255), -1)  # Este es el centro de la imagen
                cv2.circle(res, (inter_x, inter_y), 5, (0, 255, 0), -1)  # Esta es la intersección de las lineas
                # cv2.imshow("img",img)
                # cv2.imshow("mask",opening)
                cv2.imshow("res", res)
                error_ant = error

                time.sleep(0.01)
                # cv2.waitKey(0)


        else:
            print("No hay 1 linea")
    else:
        print("Ninguna linea")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
