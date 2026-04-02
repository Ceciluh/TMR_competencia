import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap   = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow("prueba", model(frame, conf=0.30, verbose=False)[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()