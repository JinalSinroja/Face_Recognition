# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob

files=[]
img_array = []
for filename in glob.glob('C:\\Users\\jinal\\Documents\\Face Recognition\\srkAnswers\\*.jpg'):
  files.append(filename)
tiles = sorted(files)
# print(tiles)
# files.sort(key = lambda x: x[5:-4])
# files = set(files)
for filename in tiles:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('C:\\Users\\jinal\\Documents\\Face Recognition\\Videos\\OutputVideos\\srk.avi',cv2.VideoWriter_fourcc(*'DIVX'), 6, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
