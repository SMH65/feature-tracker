import cv2

src = cv2.imread("SSH_test_img.jpg", cv2.IMREAD_COLOR)

height, width, channel = src.shape
matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
dst = cv2.warpAffine(src, matrix, (width, height))

cv2.imwrite('rotated.jpg', dst)