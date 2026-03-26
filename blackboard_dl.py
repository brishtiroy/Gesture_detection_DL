import cv2
import numpy as np
import collections
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2


# -----------------------------
# 1. THE DEEP LEARNING ARCHITECTURE
# -----------------------------
def build_gesture_model(input_shape=(10, 224, 224, 3), num_classes=3):
    """
    CNN-GRU Hybrid:
    - CNN: Extracts 'What' (Hand shape/position)
    - GRU: Extracts 'When' (Movement/Intent over 10 frames)
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        TimeDistributed(base_model, input_shape=input_shape),
        TimeDistributed(GlobalAveragePooling2D()),
        GRU(64, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        # Classes: 0=Neutral, 1=Draw (Active), 2=Erase
        Dense(num_classes, activation='softmax')
    ])
    return model


# -----------------------------
# 2. BLACKBOARD ENGINE
# -----------------------------
class DL_Blackboard:
    def __init__(self):
        self.canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        self.color = (255, 255, 255)
        self.thick = 5
        self.prev_pos = None
        # Temporal Buffer for 10-frame snippets
        self.frame_buffer = collections.deque(maxlen=10)

    def process_inference(self, frame):
        # Preprocess for CNN
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
        self.frame_buffer.append(img_resized)

        if len(self.frame_buffer) < 10:
            return 0, (0, 0)

        # In a live app: prediction = model.predict(np.expand_dims(self.frame_buffer, axis=0))
        # Here we simulate the DL output: Class 1 (Drawing) and Centroid Tracking
        gesture_class = 1
        # Centroid would be extracted from the CNN's Heatmap layer
        current_pos = (400, 300)

        return gesture_class, current_pos


# -----------------------------
# 3. MAIN EXECUTION
# -----------------------------
def run_pure_dl_blackboard():
    bb = DL_Blackboard()
    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 600)

    print("DL Blackboard Initialized. System: CNN + GRU.")

    while True:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)

        # --- DL PIPELINE ---
        gesture_id, pos = bb.process_inference(frame)

        # Action Logic based on DL Classification
        if gesture_id == 1:  # 'Draw' intent recognized by GRU
            if bb.prev_pos is not None:
                cv2.line(bb.canvas, bb.prev_pos, pos, bb.color, bb.thick)
            bb.prev_pos = pos
        elif gesture_id == 2:  # 'Erase' intent
            cv2.circle(bb.canvas, pos, 30, (0, 0, 0), -1)
            bb.prev_pos = None
        else:
            bb.prev_pos = None

        # --- UI RENDERING ---
        # Merge canvas with live camera (Glass Blackboard effect)
        combined = cv2.addWeighted(frame, 0.4, bb.canvas, 0.6, 0)

        # Visual Pointer (The 'AI Cursor')
        cv2.circle(combined, pos, 8, (0, 255, 0), -1)

        cv2.imshow("100% DL Virtual Blackboard", combined)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pure_dl_blackboard()