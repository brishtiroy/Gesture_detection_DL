from tensorflow.keras.models import load_model
import cv2
import numpy as np
from collections import deque

model = load_model("cnn_lstm_model.h5")

cap = cv2.VideoCapture(0)
seq = deque(maxlen=10)

equation = ""

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    img = cv2.resize(frame, (64,64))
    seq.append(img)

    if len(seq) == 10:
        input_data = np.array(seq) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        pred = model.predict(input_data, verbose=0)
        gesture = np.argmax(pred)

        # Map gestures to calculator
        if gesture == 0:
            equation += "1"
        elif gesture == 1:
            equation += "+"
        elif gesture == 2:
            equation = str(eval(equation)) if equation else ""

    cv2.putText(frame, equation, (50,100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

    cv2.imshow("Calculator DL", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()