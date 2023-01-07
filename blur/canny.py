import cv2
import numpy as np
from matplotlib import pyplot as plt

sigma=0.43

gray = cv2.imread('max.jpg', cv2.IMREAD_COLOR)
img = cv2.GaussianBlur(gray, (7, 7), 0)

v = np.median(img)

lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edges = cv2.Canny(img, lower, upper)

#plt.subplot(121)
#plt.imshow(img,cmap = 'gray')
#plt.title('Original Image') 
#plt.xticks([]), plt.yticks([])
#plt.subplot(122)
#plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image')
#plt.xticks([]) 
#plt.yticks([])

#plt.show()

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.namedWindow('Edges',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Edges', 600,600)

cv2.imshow("image", img)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()