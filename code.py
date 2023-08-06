#8.1
import cv2
import numpy as np

path = r'C:\Users\Naeem\Desktop\Jahanzeb\DIP\DIP Lab\images\333.jpg'
img = cv2.imread(path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (0, 0), None, .25, .25)

gaussianBlurKernel = np.array(([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), np.float32)/9
meanBlurKernel = np.ones((3, 3), np.float32)/9

gaussianBlur = cv2.filter2D(src=img, kernel=gaussianBlurKernel, ddepth=-1)
meanBlur = cv2.filter2D(src=img, kernel=meanBlurKernel, ddepth=-1)

horizontalStack = np.concatenate((img, gaussianBlur, meanBlur), axis=1)

cv2.imshow("2D Convolution Example", horizontalStack)

cv2.waitKey(0)
cv2.destroyAllWindows()





#8.2
import cv2
import numpy as np

path = r'C:\Users\Naeem\Desktop\Jahanzeb\DIP\DIP Lab\images\SP.jpg'
img = cv2.imread(path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (0, 0), None, .25, .25)

medianBlur = cv2.medianBlur(img, 5)

horizontalStack = np.concatenate((img, medianBlur), axis=1)

cv2.imshow("2D Convolution Example", horizontalStack)

cv2.waitKey(0)
cv2.destroyAllWindows()






#8.3import cv2
import numpy as np

path = r'C:\Users\Naeem\Desktop\Jahanzeb\DIP\DIP Lab\images\moon.jpg'
img = cv2.imread(path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (0, 0), None, .25, .25)

def Mylaplacian(img):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    dst = cv2.filter2D(img, -1, kernel)
    return dst

img2 = Mylaplacian(img)
cv2.imshow("Laplacian",img2)

Y = np.array([[-2,-5,-2], [0,0,0], [2,5,2]], np.float32)
X = np.array([[2,0,-2], [5,0,-5], [2,0,-2]], np.float32)

sobelX = cv2.filter2D(img,-1,X)
sobelY = cv2.filter2D(img,-1,Y)

final = abs(sobelY)+abs(sobelX)

horizontalStack = np.concatenate((sobelX, sobelY, final), axis=1)
cv2.imshow("Sobel", horizontalStack)

cv2.waitKey(0)
cv2.destroyAllWindows()







#9.1
import cv2
import numpy as np

img = cv2.imread("lines.jpeg")
img = cv2.resize(img, (0, 0), None, .5, .5)

HorizKn = np.array(([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]), np.float32)
Ng45Kn = np.array(([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]), np.float32)
VertiKn = np.array(([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]), np.float32)
Ps45Kn = np.array(([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]), np.float32)

HorizBlur = cv2.filter2D(src=img, kernel=HorizKn, ddepth=-1)
Ng45Blur = cv2.filter2D(src=img, kernel=Ng45Kn, ddepth=-1)
VertiBlur = cv2.filter2D(src=img, kernel=VertiKn, ddepth=-1)
Ps45Blur = cv2.filter2D(src=img, kernel=Ps45Kn, ddepth=-1)

horizStck = np.concatenate((img, HorizBlur, Ng45Blur, VertiBlur, Ps45Blur), axis=1)
cv2.imwrite("Output.jpg", horizStck)
cv2.imshow("Edge Detection", horizStck)

cv2.waitKey(0)
cv2.destroyAllWindows()






#9.2
import cv2
import numpy as np

img = cv2.imread("curves.jpeg")
img = cv2.resize(img, (0, 0), None, .25, .25)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

HorizKn = np.array(([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]), np.float32)
Ng45Kn = np.array(([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]), np.float32)
VertiKn = np.array(([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]), np.float32)
Ps45Kn = np.array(([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]), np.float32)

HorizBlur = cv2.filter2D(src=img, kernel=HorizKn, ddepth=-1)
Ng45Blur = cv2.filter2D(src=img, kernel=Ng45Kn, ddepth=-1)
VertiBlur = cv2.filter2D(src=img, kernel=VertiKn, ddepth=-1)
Ps45Blur = cv2.filter2D(src=img, kernel=Ps45Kn, ddepth=-1)

horizStck = np.concatenate((img, HorizBlur, Ng45Blur, VertiBlur, Ps45Blur), axis=0)
cv2.imwrite("Output.jpg", horizStck)
cv2.imshow("Edge Detection", horizStck)






#9.3
import cv2
import numpy as np

img = cv2.imread("tiger.jpg")
img = cv2.resize(img, (0, 0), None, .5, .5)
HorizKn= np.array(([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]), np.float32)
Ng45Kn = np.array(([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]), np.float32)
VertiKn = np.array(([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]), np.float32)
Ps45Kn = np.array(([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]), np.float32)
SobelX = cv2.Sobel(img, cv2.CV_32F, dx=1, dy=0, ksize=3)
SobelY = cv2.Sobel(img, cv2.CV_32F, dx=0, dy=1, ksize=3)

cv2.imshow("sobelX", SobelX)
cv2.imshow("sobelY", SobelY)
SobelXY = cv2.Sobel(img, cv2.CV_32F, dx=1, dy=1, ksize=3)
cv2.imshow("sobelXY", SobelXY)
cv2.waitKey(0)
cv2.destroyAllWindows()






#12.1
import cv2 as cv
import numpy as np
image=cv.imread('coins.JPG',0)
kernel=np.ones((15,15), np.uint8)
closing = cv.morphologyEx(image,cv.MORPH_CLOSE,kernel)
kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
erosion=cv.erode(closing,kernel,iterations=1)
boundaryImage=closing-erosion
cv.imshow("original",boundaryImage)
cv.waitKey(0)
cv.destroyAllWindows()







#12.2
import cv2 as cv
import numpy as np
image=cv.imread('lines.jpg',0)
cv.imshow("lines",image)
kernel=np.ones((15,2),np.uint8)
#kernal1=np.ones(3,13)
verticalLines=cv.erode(image,kernel,iterations=1)
#horizontalLines=cv.erode(image,kernal1,iterations=1)
cv.imshow("vertical Lines",verticalLines)
#cv.imshow("horizontal Lines",horizontalLines)
cv.waitKey(0)


import cv2 as cv
import numpy as np
image=cv.imread('lines.jpg',0)
cv.imshow("lines",image)
#kernel=np.ones((15,2),np.uint8)
kernal1=np.ones((3,17),np.uint8)
#verticalLines=cv.erode(image,kernel,iterations=1)
horizontalLines=cv.erode(image,kernal1,iterations=1)
#cv.imshow("vertical Lines",verticalLines)
cv.imshow("horizontal Lines",horizontalLines)
cv.waitKey(0)






#12.3
import numpy as np
import cv2

img = cv2.imread('text.png',0)
kernel=np.ones((3,5),np.uint8)
words=cv2.erode(img,kernel,iterations=1)

totalLabels,labels,stats,centroids=cv2.connectedComponentsWithStats(~words,8,cv2.CV_32S)
print("total words are:",totalLabels-1)

colors=np.random.randint(0,255,size=(totalLabels,3),dtype=np.uint8)
colors[0]=[0,0,0]
colored_components=colors[labels]

cv2.imshow('output',~words )
cv2.imshow('colored image',colored_components)
cv2.waitKey(0)






#12.4
import cv2
# Read image from which text needs to be extracted
img = cv2.imread("my_image.JPG")

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Performing binarization through threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

# Creating a copy of image
im2 = img.copy()  # Why did we use the copy method instead of just assigning like im2 = img. Any idea?
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Cropping the text block for giving input to OCR
    cropped = im2[y:y + h, x:x + w]

cv2.imshow('output', im2)
cv2.waitKey(0)







