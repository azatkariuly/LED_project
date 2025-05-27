import cv2
import numpy as np

# Load the image
image = cv2.imread("modules/res_red.png")  # Change to your image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Edge Detection (Canny)
edges = cv2.Canny(blurred, threshold1=50, threshold2=100)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw detected contours
output = image.copy()
# cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
for contour in contours:
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    if 12 < h < 20:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), -1)

# Display results
cv2.imshow("Edges of Broken Color", edges)
cv2.imshow("Detected Contours", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
