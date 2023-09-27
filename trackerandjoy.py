import cv2
from utils import *
import vgamepad as vg
import time

w, h = 360, 240

gamepad = vg.VX360Gamepad()

cap = cv2.VideoCapture(0)

gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
gamepad.update()
time.sleep(2.5)

while True:
    ret, frame = cap.read()    
    frame, info, coordinates = findFace(frame)
    cv2.imshow('Image', frame)
    
    if coordinates is not None:
        x, y = coordinates
        # print(f"Coordinates (x, y): ({x:.2f}, {y:.2f})")
        gamepad.left_joystick_float(x_value_float=x, y_value_float=y)
        gamepad.update()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
