#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import pyurg
import math
import matplotlib.pyplot as plt

X = []
Y = []
# For initializing.
urg = pyurg.UrgDevice()
plt.ion()
port = "/dev/ttyACM0"

# Connect to the URG device.
# If could not conncet it, get False from urg.connect()
if not urg.connect(port = port,baudrate=115200, timeout=0.1):
    print 'Could not connect.'
    exit()

# Get length datas and timestamp.
# If missed, get [] and -1 from urg.capture()

pitch = 0.36
angle = (pitch*math.pi)/180

#time.sleep(2)
data, timestamp = urg.capture()
print(timestamp)
# Print lengths.
i=0
j=0
for length in data:

    if (i > 43):
        x = length*math.sin(angle)
        y = length*math.cos(angle)
        X[j] = x
        Y[j] = y
        j = j+1
        print (x,y)

    angle = angle + angle
    i=i+1
print("i",i)

plt.plot([X], [Y], 'ro')
# plt.axis([0, 6, 0, 20])
plt.show()