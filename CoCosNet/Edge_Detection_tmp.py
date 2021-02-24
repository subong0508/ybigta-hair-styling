import cv2
import numpy
src = cv2.imread("imgs/mymodel/source_img/00001.jpg")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(src, 100, 255)
canny = numpy.array(canny)
cv2.imwrite('/nfs/home/ryan0507/cocosnet/result.png',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()