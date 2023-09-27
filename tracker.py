import cv2
from utils import *

w, h = 360, 240

cap = cv2.VideoCapture(0)  # Open the default camera (0 or -1)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    
    # Step 2: Find faces in the frame
    frame, info, coordinates = findFace(frame)

    # Display the image with face detection rectangles
    cv2.imshow('Image', frame)
    
    # Print the coordinates to the terminal
    if coordinates is not None:
        x, y = coordinates
        print(f"Coordinates (x, y): ({x:.2f}, {y:.2f})")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
