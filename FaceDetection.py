import cv2

def Detect(img, img_color):  
    cascade_alt = cv2.CascadeClassifier()
    cascade_alt.load("/home/neha/workspace/face_anonymization/haarcascades/haarcascade_frontalface_alt.xml")
    rects = cascade_alt.detectMultiScale(img, 1.1, 3, cv2.cv.CV_HAAR_SCALE_IMAGE, (3,3))
    
    #cascade_alt2 = cv2.CascadeClassifier()
    #cascade_alt2.load("/home/kostas/PythonProjects/BlurFaces/haarcascades/haarcascade_frontalface_alt2.xml")
    #rects_add = cascade_alt2.detectMultiScale(img, 1.1, 3, cv2.cv.CV_HAAR_SCALE_IMAGE, (3,3))
    #if len(rects_add) != 0:
    #    rects = np.concatenate((rects, rects_add), axis=0)     
    
    #cascade_alt_tree = cv2.CascadeClassifier()
    #cascade_alt_tree.load("/home/kostas/PythonProjects/BlurFaces/haarcascades/haarcascade_frontalface_alt_tree.xml")
    #rects_add = cascade_alt_tree.detectMultiScale(img, 1.1, 3, cv2.cv.CV_HAAR_SCALE_IMAGE, (3,3))
    #if len(rects_add) != 0:
    #    rects = np.concatenate((rects, rects_add), axis=0)     
    
    #cascade_default = cv2.CascadeClassifier()
    #cascade_default.load("/home/kostas/PythonProjects/BlurFaces/haarcascades/haarcascade_frontalface_default.xml")
    #rects_add = cascade_default.detectMultiScale(img, 1.1, 3, cv2.cv.CV_HAAR_SCALE_IMAGE, (3,3))
    #if len(rects_add) != 0:
    #    rects = np.concatenate((rects, rects_add), axis=0) 

    
    if len(rects) != 0:
        rects[:, 2:] += rects[:, :2]
    box(rects, img)    
   
    return rects

def box(rects, img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

def Blur(rects,img_color):
    for x1, y1, x2, y2 in rects:
        roi = img_color[y1:y2, x1:x2, :]
        roi = cv2.blur(roi,(20,20))
        img_color[y1:y2, x1:x2, :] = roi
    return img_color
    
    
    
    
    
