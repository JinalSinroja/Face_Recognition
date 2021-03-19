# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:14:11 2020

@author: jinal
"""

from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import cv2

import face_recognition
from PIL import Image, ImageDraw

import numpy as np
import glob



def faceDetectionMtcnn(filename):
  # filename = '/content/drive/My Drive/unknown/aamieside2.jpg'
  # load image from file
  pixels = pyplot.imread(filename)
  # create the detector, using default weights
  detector = MTCNN()
  # detect faces in the image
  faces = detector.detect_faces(pixels)
  # print(faces[0]['box'][0])
  face_locations = []
  # print(faces)
  for i in range(len(faces)):
    temp = faces[i]['box']
    t1=faces[i]['box'][0]
    t2=faces[i]['box'][1]
    t3=faces[i]['box'][2]
    t4=faces[i]['box'][3]
    faces[i]['box'][0]=int(t2)
    faces[i]['box'][1]=int(t1+t3)
    faces[i]['box'][2]=int(t2+t4)
    faces[i]['box'][3]=int(t1)
    face_locations.append(tuple(faces[i]['box']))
    
  print(face_locations)
  return face_locations
  # display faces on the original image



#########################
# run this
#########################
# method for creating encodings of known faces


image_of_aamir = face_recognition.load_image_file('C:\\Users\\jinal\\Documents\\Face Recognition\\Project\\images\\----\\face1.jpg')
filename = 'C:\\Users\\jinal\\Documents\\Face Recognition\\Project\\images\\----\\srk.jpg'
face_locations= faceDetectionMtcnn(filename)
srk_face_encoding = face_recognition.face_encodings(image_of_aamir, face_locations)[0]


image_of_aamir = face_recognition.load_image_file('C:\\Users\\jinal\\Documents\\Face Recognition\\Project\\images\\---\\face2.jpg')
filename = 'C:\\Users\\jinal\\Documents\\Face Recognition\\Project\\images\\----\\shahrukhside.jpg'
face_locations= faceDetectionMtcnn(filename)
srk_side1_face_encoding = face_recognition.face_encodings(image_of_aamir, face_locations)[0]
# srk_face_encoding = face_recognition.face_encodings(image_of_aamir)[0]
print("rinting srks encoding")

#  Create arrays of encodings and names
known_face_encodings = [
  srk_face_encoding,
  srk_side1_face_encoding
]

known_face_names = [
  "----",
  "----"
]

   

ourcount = 0
# Load test image to find faces in
def identify(filename):
  
  print("function start")
  matches=[]
  test_image = face_recognition.load_image_file(filename)
  test_image_filename = filename
  output= test_image.copy()
  # Find faces in test image
  # face_locations = face_recognition.face_locations(test_image)
  face_locations= faceDetectionMtcnn(test_image_filename)
  face_encodings = face_recognition.face_encodings(test_image, face_locations)
  # print(face_encodings)

  # create a list new list of only box from faces wala dictionary


  # Convert to PIL format
  pil_image = Image.fromarray(test_image)
  # print(pil_image)

  # Create a ImageDraw instance
  draw = ImageDraw.Draw(pil_image)

  # Loop through faces in test image
  for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
    print( "List of matches: ",matches)
    name = "Unknown Person"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    print("index of best match: ", best_match_index)
    if matches[best_match_index]:
      name = known_face_names[best_match_index]


    # Draw box
    draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))
    # Draw label
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))


  del draw
  # Display image
  

  # Save image
  
  for i in matches:

    if i:
      global ourcount;
      ourcount = format(ourcount, '05d')
      print("going to save the image")
      # pil_image.save('/content/drive/My Drive/answerfolder/identifys%s.jpg' % ourcount)
      pil_image.save('C:\\Users\\jinal\\Documents\\Face Recognition\\Project\\images\\----\\FramesAnswer\\identifys%s.jpg' % ourcount)
      # file= '/content/answersaamir2/identifys%s.jpg' % ourcount
      # cv2_imshow(file)
      ourcount = int(ourcount) + 1

      print("theres true")

#cv2.destroyAllWindows()  

cap = cv2.VideoCapture("C:\\Users\\jinal\\Documents\\Face Recognition\\Project\\images\\----\\aamir.mp4")
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
  #  size variable is initialized to be used in writer object
size = (frame_width, frame_height) 

#########################################################################
count = 0
# frameRate = 0.1 #frame rate
x=1
while(cap.isOpened()):
    print(x)
    x +=1
    # frameId = cap.get(1) #current frame number
    ret, frame = cap.read()


    if ret:
      # filename = 'frame{:d}.jpg'.format(count)
      # filename ="frames%d.jpg" % count;
      cv2.imwrite('C:\\Users\\jinal\\Documents\\Face Recognition\\Project\\images\\-----\\Frames\\frames{:d}.jpg'.format(count), frame)
      # count += 30 # i.e. at 30 fps, this advances one second
      count += 6
      cap.set(1, count)
      
    else:
        cap.release()
        break

    
cap.release()

for filename in glob.glob('C:\\Users\\jinal\\Documents\\Face Recognition\\Project\\images\\-----\\FrameAnswer\\*.jpg'):
  print("alling iddentigy")
  identify(filename)

files=[]
img_array = []
for filename in glob.glob('C:\\Users\\jinal\\Documents\\Face Recognition\\Project\\images\\-----\\FrameAnswer\\*.jpg'):
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


out = cv2.VideoWriter('C:\\Users\\jinal\\Documents\\Face Recognition\\Project\\images\\-----\\downloadVideo.avi',cv2.VideoWriter_fourcc(*'DIVX'), 6, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

