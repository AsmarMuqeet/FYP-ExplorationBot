LIDAR_DEVICE            = '/dev/ttyUSB0'
MIN_SAMPLES   = 180

from rplidar import RPLidar as Lidar
import io
import os
import socket
import struct
import pickle
import sys
import time

if __name__ == '__main__':

    # Connect to Lidar unit
    try:
        lidar = Lidar(LIDAR_DEVICE)
        client_socket = socket.socket()
        client_socket.settimeout(5)
        client_socket.connect(('localhost', 3333))
        # Create an iterator to collect scan data from the RPLidar
        iterator = lidar.iter_scans()
        next(iterator)
        #maxlist = []

        while True:

            # Extract (quality, angle, distance) triples from current scan
            items = [item for item in next(iterator)]
            #print(items)
            data=pickle.dumps(items)
            #client_socket.send(pickle.dumps(sys.getsizeof(data)))
            #maxlist.append(sys.getsizeof(data))
            #time.sleep(0.5)
            client_socket.send(data)
    except Exception as e:
    # Shut down the lidar connection
        print(e)
        #print(max(maxlist))
        lidar.stop()
        lidar.disconnect()
        client_socket.close()

