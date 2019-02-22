LIDAR_DEVICE            = '/dev/ttyUSB0'
MIN_SAMPLES   = 180
iterator = 0

from rplidar import RPLidar as Lidar
import io
import os
import socket
import struct
import pickle
import sys
import time
import json
from flask import Flask

app = Flask(__name__)

@app.route('/')
def getdata():
    global iterator
    items = [item for item in next(iterator)]
    return json.dumps(items)

if __name__ == '__main__':
    # Connect to Lidar unit
    lidar = Lidar(LIDAR_DEVICE)
    iterator = lidar.iter_scans()
    next(iterator)
    app.run()
