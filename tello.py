from djitellopy import Tello
import time

test = Tello()
test.connect()

print(test.get_battery())






