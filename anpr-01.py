import cv2
import pytesseract
import imutils

# Load image
image = cv2.imread('datasets/number-plates/1.jpeg')

# Resize image
image = imutils.resize(image, width=500)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply image processing techniques
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

# Find contours
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Detect number plate
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ratio = w / float(h)
        
        if 2.5 < ratio < 4.5:
            number_plate = gray[y:y + h, x:x + w]
            break

# Apply OCR
text = pytesseract.image_to_string(number_plate, lang='eng',
                                   config='--psm 11')

# Display results
cv2.imshow('Number Plate', number_plate)
print('Number Plate:', text.strip())
cv2.waitKey(0)
