import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
image = cv2.imread("lab6.jpg")
if image is None:
    print("Error: Could not read image.")
    exit()

max_size = 800
height, width, _ = image.shape
scale = max_size / max(height, width)
new_width = int(width * scale)
new_height = int(height * scale)
image = cv2.resize(image, (new_width, new_height))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

for (x, y, w, h) in boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("HOG Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
