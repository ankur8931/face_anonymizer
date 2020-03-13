import cv2
import numpy as np
import FaceDetection   


cap = cv2.VideoCapture('/home/neha/workspace/face_anonymization/face-demographics-walking.mp4')
width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

img_array = []
count = 0
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = FaceDetection.Detect(gray, frame)
    out_img = FaceDetection.Blur(rects, frame)

    cv2.imshow('out_img',out_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    filename = "results/out_img%d.jpg" % count
    cv2.imwrite(filename, out_img)
    
    count = count + 1
    img_array.append(out_img)

fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('results/output_video.avi',fourcc, 1, (width,height))

for i in range(len(img_array)):
    out.write(img_array[i])

cap.release()
cv2.destroyAllWindows()
out.release()



