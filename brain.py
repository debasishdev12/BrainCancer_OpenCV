import cv2
import numpy as np

img = cv2.imread("brain1.jpg")

newX,newY = 600,480
krnl = np.ones((15,15),np.uint8)
img1 = cv2.resize(img,(newX,newY))
hsv = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(hsv,(55,55),0)
_,thrsh = cv2.threshold(hsv,120,255,cv2.THRESH_BINARY)
mask1 = cv2.morphologyEx(thrsh,cv2.MORPH_OPEN,krnl)
mask2 = cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,krnl)
_,inv = cv2.threshold(mask2,127,255,cv2.THRESH_BINARY_INV)
dst = cv2.bitwise_or(img1,img1,mask=inv)

cv2.imshow("Input",img1)
cv2.imshow("Output",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

