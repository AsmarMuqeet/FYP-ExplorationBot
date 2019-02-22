import socket
import json
import socket
import json
import numpy as np
from scipy.misc import imsave
import ast
import pandas as pd
import cv2

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('192.168.0.193', 1236))
s.listen(1)
conn, addr = s.accept()
b = b''
print 'connected'
k = 1
kernel = np.ones((11,15),np.float32)/31
kernel2 = np.ones((7,7), np.uint8)
while 1:
    tmp = conn.recv(1024)
    s = ast.literal_eval(tmp)
    print s
    data = pd.DataFrame(s["2"],columns=['row','col'],dtype='int16')
    data2 = pd.DataFrame(s["1"],columns=['row','col'],dtype='int16')
    maxrow = max(data['row'].max(),data2['row'].max())
    maxcol = max(data['col'].max(),data2['col'].max())
    minrow = min(data['row'].min(),data2['row'].min())
    mincol = min(data['col'].min(),data2['col'].min())
    print "maxrow :",maxrow," minrow :",minrow
    print "maxcol :",maxcol," mincol :",mincol
    K = np.zeros((maxrow+20,maxcol+20,3),dtype='int16')
    K[[data['row'].values],[data['col'].values]] =[255,0,0]
    K[[data2['row'].values],[data2['col'].values]] =[0,0,255]
    RA = K[minrow:maxrow+1,mincol:maxcol+1,:]	
    img_dilation = cv2.dilate(RA, kernel2, iterations=1)
    dst = cv2.filter2D(img_dilation,-1,kernel,(100,9))
    imsave('map.png',dst)
    k+=1
    b += tmp
d = json.loads(b.decode('utf-8'))
