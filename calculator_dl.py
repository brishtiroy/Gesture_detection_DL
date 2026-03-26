import cv2
import numpy as np
import collections
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2


# -----------------------------
# 1. THE DEEP LEARNING MODEL
# -----------------------------
def build_dl_model(input_shape=(15, 224, 224, 3), num_classes=3):
    """
    Completely DL-based: TimeDistributed CNN + LSTM
    Input: 15 consecutive frames (a video snippet)
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model = Sequential([
        # CNN handles the 'Visual' part of each frame
        TimeDistributed(base_model, input_shape=input_shape),
        TimeDistributed(GlobalAveragePooling2D()),

        # LSTM handles the 'Time' part (the movement)
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')  # [0: Neutral, 1: Point, 2: Click]
    ])
    return model


# -----------------------------
# 2. CALCULATOR ENGINE
# -----------------------------
class VirtualCalc:
    def __init__(self):
        self.eq = ""
        self.buttons = [
            ("7", (350, 200)), ("8", (480, 200)), ("9", (610, 200)), ("/", (740, 200)),
            ("4", (350, 310)), ("5", (480, 310)), ("6", (610, 310)), ("*", (740, 310)),
            ("1", (350, 420)), ("2", (480, 420)), ("3", (610, 420)), ("-", (740, 420)),
            ("0", (350, 530)), ("C", (480, 530)), ("=", (610, 530)), ("+", (740, 530))
        ]

    def update(self, val):
        if val == "=":
            try:
                self.eq = str(eval(self.eq))
            except:
                self.eq = "Error"
        elif val == "C":
            self.eq = ""
        else:
            self.eq += val


# -----------------------------
# 3. MAIN RUNTIME LOOP
# -----------------------------
def run_dl_calculator():
    calc = VirtualCalc()
    cap = cv2.VideoCapture(0)

    # DL Buffer: We store the last 15 frames to feed into the LSTM
    frame_buffer = collections.deque(maxlen=15)

    # Load your trained DL weights here
    # model = load_model('hand_gesture_lstm.h5')

    print("DL Calculator Started. Using CNN+LSTM Pipeline...")

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)

        # PRE-PROCESSING FOR DL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
        frame_buffer.append(img_resized)

        current_gesture = 0  # Default: Neutral

        # DL INFERENCE
        if len(frame_buffer) == 15:
            # Prepare the 4D input: (Batch, Frames, Width, Height, Channels)
            input_data = np.expand_dims(np.array(frame_buffer), axis=0)

            # prediction = model.predict(input_data, verbose=0)
            # current_gesture = np.argmax(prediction)

            # For demonstration, we'll assume gesture 2 is 'Click'
            # (In a live app, this replaces all the 'pinch distance' code)

        # UI DRAWING
        # Result Display
        cv2.rectangle(img, (350, 50), (860, 150), (255, 255, 255), -1)
        cv2.putText(img, calc.eq, (370, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

        # Draw Buttons and Detect Interaction
        # (Using center of screen as 'DL Pointer' for this logic)
        pointer_x, pointer_y = 640, 480

        for text, pos in calc.buttons:
            x, y = pos
            w, h = 110, 90

            is_hover = x < pointer_x < x + w and y < pointer_y < y + h
            color = (0, 255, 0) if is_hover else (200, 200, 200)

            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            cv2.putText(img, text, (x + 30, y + 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            # If DL Model detects 'CLICK' gesture (2) while hovering
            if is_hover and current_gesture == 2:
                calc.update(text)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), -1)  # Visual feedback

        cv2.imshow("100% DL Hand Calculator", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_dl_calculator()