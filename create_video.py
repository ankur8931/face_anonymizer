import os
import glob
import cv2

os.chdir("/home/neha/workspace/face_anonymization/results")
cap = cv2.VideoCapture('/home/neha/workspace/face_anonymization/face-demographics-walking.mp4')
#NEHA
width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
#output_video = cv2.VideoWriter("results/output_video1.avi", 0, 25.0, (width, height))

img_array = []
for file in glob.glob("*.jpg"):
	img1 = cv2.imread(file)
	#out = cv2.VideoWriter('out.avi',-1,1,(width,height))
	#out.write(img1)
	#out = cv2.VideoWriter("results/output_video1.avi", 0, 25.0, (width, height))
	img_array.append(img1)
 
fourcc = cv2.cv.CV_FOURCC(*'XVID') 
out = cv2.VideoWriter('project.avi',fourcc, 15, (width,height))
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
