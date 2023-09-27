import cv2
import numpy as np

def findFace(frame):
    # Load the face detection classifier
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Create a mask of the same size as the frame
    mask = np.ones_like(frame) * 255  # Initialize the mask as white

    # Convert the frame to grayscale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(frameGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    coordinates = None  # Initialize coordinates to None

    # Draw a circle at the center of the screen
    cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 5, (0, 255, 0), -1)


    if len(faces) > 0:
        # Choose the largest face (assuming the largest face corresponds to the person of interest)
        max_area = 0
        max_face = faces[0]
        for (x, y, w, h) in faces:
            if w * h > max_area:
                max_area = w * h
                max_face = (x, y, w, h)

        x, y, w, h = max_face

        # Calculate the center of the detected face
        cx = x + w // 2
        cy = y + h // 2

        # Calculate the coordinates relative to the center of the screen and normalize them
        rel_x = (cx - frame.shape[1] // 2) / (frame.shape[1] // 4)  # Adjust the denominator for scaling
        rel_y = -(cy - frame.shape[0] // 2) / (frame.shape[0] // 4)  # Adjust the denominator for scaling

        # Clamp the x and y coordinates to the range of -1 to 1
        rel_x = max(-1, min(1, rel_x))
        rel_y = max(-1, min(1, rel_y))

        # Calculate the position of the nose
        nose_x = cx - x
        nose_y = cy - y

        coordinates = (rel_x, rel_y)  # Store the coordinates

        # Draw a dot on the nose
        cv2.circle(frame, (x + nose_x, y + nose_y), 3, (255, 0, 0), -1)

    # Create a circle with a radius of 1 centered at (0, 0) on the mask
    circle_radius = 1
    circle_center = (mask.shape[1] // 2, mask.shape[0] // 2)
    cv2.circle(mask, circle_center, int(circle_radius * mask.shape[1] // 4), (0, 0, 0), -1)  # Adjust the radius for scaling

    # Use the mask to make the area outside the circular region white
    frame = cv2.bitwise_or(frame, mask)

    if len(faces) > 0:
        # Display the normalized relative coordinates in the upper portion of the frame
        text = f"Coordinates (x, y): ({rel_x:.2f}, {rel_y:.2f})"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_x = frame.shape[1] // 2 - text_size[0] // 2
        text_y = 30
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    else:
        # No faces detected, display "Reposition yourself" in red in the upper portion of the frame
        text = "Reposition yourself"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_x = frame.shape[1] // 2 - text_size[0] // 2
        text_y = 30
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return frame, faces, coordinates  # Return coordinates along with the frame and faces
