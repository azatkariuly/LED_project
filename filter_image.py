import cv2

# Global variables to store the starting and ending points of the rectangle
start_point = None
end_point = None
drawing = False  # Flag to indicate if the mouse is being dragged

def detect_billboard(frame):
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Step 3: Perform edge detection
    edges = cv2.Canny(blurred, 100, 160)

    cv2.imshow("Edges", edges)

    # Step 4: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)

    # Step 5: Loop through contours and filter for rectangles
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the approximated contour has 4 points (rectangle)
        if len(approx) == 4:
            # Check the contour's bounding box size and aspect ratio
            # print('sujka', approx)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h

            # Define criteria for a billboard
            # if 0.8 <= aspect_ratio <= 1.2 and w > 50 and h > 50:  # Adjust as needed
                # Draw the detected rectangle on the frame
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                # return frame, approx

    return frame, None

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, frame
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button pressed
        start_point = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:  # Mouse is moving
        if drawing:
            end_point = (x, y)
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("Frame", temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:  # Left mouse button released
        end_point = (x, y)
        drawing = False
        frame = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        # cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
        frame, detected_rectangle = detect_billboard(frame)
        cv2.imshow("Frame", frame)

# Load an image or video frame
frame = cv2.imread("stretched_image.png")  # Replace with your frame source
if frame is None:
    print("Error: Could not load image.")
    exit()

# cv2.imshow("Frame", frame)

frame, approx = detect_billboard(frame)

# cv2.imshow("Edges", frame)

# Set the mouse callback function
# cv2.setMouseCallback("Frame", draw_rectangle)

# Wait for a key press and clean up
cv2.waitKey(0)
cv2.destroyAllWindows()
