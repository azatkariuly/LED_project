# import cv2

# # Your RTSP URL (replace with your actual RTSP stream URL)
# # rtsp_url = "http://61.84.167.193/wmf/index.html#/uni/channel"

# # Wisenet RTSP URL with credentials (replace with your actual details)
# username = "admin"
# password = "postech!8880"
# ip_address = "61.84.167.193"
# port = "554"  # Usually 554 for RTSP
# path = "profile1/media.smp"  # Usually something like /stream1

# # Construct the RTSP URLz
# rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}/{path}"
# # rtsp_url = 'rtsp://admin:admin1357!@142.10.7.120:554/channel1'

# # Open the RTSP stream
# cap = cv2.VideoCapture(rtsp_url)

# if not cap.isOpened():
#     print("Error: Unable to connect to RTSP stream.")
# else:
#     while True:
#         ret, frame = cap.read()  # Read a frame from the stream
#         if not ret:
#             print("Error: Unable to fetch frame.")
#             break

#         print("Frame fetched successfully.", frame.shape)

#         # Process or display the frame
#         # cv2.imshow("RTSP Stream", frame)  # Display the frame in a window
#         cv2.imwrite('tony2.jpg', frame)
#         break

#         # Break the loop if the user presses 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

import cv2

username = "admin"
password = "!noun3577825"
ip_address = "118.42.239.19"
port = "554" # Usually 554 for RTSP
path = "profile1/media.smp" # Usually something like /stream1
rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}/{path}"

# request to rtsp and save image
cap = cv2.VideoCapture(rtsp_url)

while True:
    print('ss')
    ret, frame = cap.read()
    if not ret:
        print('no frame')
        break

    cv2.imshow('ff', frame)
    # cv2.imwrite('src/images/test.jpg', frame)
cap.release()