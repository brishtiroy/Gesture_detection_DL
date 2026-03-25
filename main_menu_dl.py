import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

from calculator_dl import run_calculator
from blackboard import run_blackboard
from presentation_dl import run_presentation

model = load_model("gesture_model.h5")

cap = cv2.VideoCapture(0)

history = deque(maxlen=10)

selected = False

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    img = cv2.resize(frame, (64,64))
    img = np.expand_dims(img, axis=0) / 255.0

    pred = model.predict(img, verbose=0)
    gesture = np.argmax(pred)

    history.append(gesture)
    final = max(set(history), key=history.count)

    cv2.putText(frame, "DL GESTURE MENU", (250,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)

    cv2.putText(frame, "0 = Presentation", (200,200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, "1 = Calculator", (200,300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, "2 = Blackboard", (200,400),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Detected: {final}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    if final == 0:
        cap.release()
        cv2.destroyAllWindows()
        run_presentation()

    elif final == 1:
        cap.release()
        cv2.destroyAllWindows()
        run_calculator()

    elif final == 2:
        cap.release()
        cv2.destroyAllWindows()
        run_blackboard()

    cv2.imshow("Menu DL", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()