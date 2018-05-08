from hokuyolx import HokuyoLX
laser = HokuyoLX()
timestamp, scan = laser.get_dist() # Single measurment mode
# Continous measurment mode
for timestamp, scan in laser.iter_dist(10):
    print(timestamp)


    #th = scan->angle_min + ((float)i / 4) * (3.141592653589 / 180)


    #x = scan * sin(th);


    #y = scan * cos(th);