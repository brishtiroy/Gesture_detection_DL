import cv2
import numpy as np
import collections
import fitz
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2


# -----------------------------
# 1. DEEP LEARNING MODEL ARCHITECTURE
# -----------------------------
def build_presentation_model(input_shape=(15, 224, 224, 3), num_classes=5):
    """
    CNN-LSTM Hybrid:
    Classes: 0: Neutral, 1: Swipe Left (Prev), 2: Swipe Right (Next), 3: Point, 4: Draw
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        TimeDistributed(base_model, input_shape=input_shape),
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


# -----------------------------
# 2. PRESENTATION ENGINE
# -----------------------------
class DLPresenter:
    def __init__(self, pdf_path):
        self.pages = self.load_pdf(pdf_path)
        self.current_page = 0
        self.annotations = []
        self.current_stroke = []
        self.frame_buffer = collections.deque(maxlen=15)  # 15-frame window

    def load_pdf(self, path):
        doc = fitz.open(path)
        imgs = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return imgs

    def process_frame(self, frame):
        # DL Preprocessing
        resized = cv2.resize(frame, (224, 224)) / 255.0
        self.frame_buffer.append(resized)

        if len(self.frame_buffer) < 15:
            return 0, (0, 0)

        # Inference: gesture_id = model.predict(np.expand_dims(self.frame_buffer, axis=0))
        # Simulated Output for logic flow:
        gesture_id = 0  # 0:Neutral, 1:Prev, 2:Next, 3:Point, 4:Draw
        cursor_pos = (400, 300)  # Extracted from CNN heatmap
        return gesture_id, cursor_pos


# -----------------------------
# 3. MAIN RUNTIME
# -----------------------------
def run_dl_presentation():
    presenter = DLPresenter("file.pdf")
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)

        # 1. DL GESTURE RECOGNITION
        gesture_id, pos = presenter.process_frame(frame)

        # 2. GESTURE LOGIC
        if gesture_id == 1:  # Swipe Left -> Previous Slide
            presenter.current_page = max(0, presenter.current_page - 1)
            presenter.annotations = []
        elif gesture_id == 2:  # Swipe Right -> Next Slide
            presenter.current_page = min(len(presenter.pages) - 1, presenter.current_page + 1)
            presenter.annotations = []
        elif gesture_id == 4:  # Draw Intent
            presenter.current_stroke.append(pos)
        else:
            if presenter.current_stroke:
                presenter.annotations.append(presenter.current_stroke)
                presenter.current_stroke = []

        # 3. RENDERING
        display_img = cv2.resize(presenter.pages[presenter.current_page], (1400, 800))

        # Draw all saved annotations
        for stroke in presenter.annotations + [presenter.current_stroke]:
            for i in range(1, len(stroke)):
                cv2.line(display_img, stroke[i - 1], stroke[i], (255, 0, 0), 4)

        # Pointer logic
        if gesture_id == 3:
            cv2.circle(display_img, pos, 15, (0, 0, 255), -1)

        cv2.imshow("Pure DL Presentation", display_img)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_dl_presentation()