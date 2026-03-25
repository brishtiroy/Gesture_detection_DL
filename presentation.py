import cv2
import os
import numpy as np
import fitz
from cvzone.HandTrackingModule import HandDetector

def pdf_to_images(path):
    doc = fitz.open(path)
    imgs = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        imgs.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    doc.close()
    return imgs

def clamp_point(x, y, w, h):
    """Clamp a point to image bounds [0..w-1], [0..h-1]."""
    return max(0, min(x, w - 1)), max(0, min(y, h - 1))

def run_presentation():
    width, height = 1400, 800
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file.pdf")
    images = pdf_to_images(pdf_path)
    if not images:
        print("No pages found in", pdf_path)
        return

    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    detector = HandDetector(detectionCon=0.75, maxHands=1)

    imgNumber = 0
    buttonPressed = False
    buttonCounter = 0
    buttonDelay = 20

    # ---- FIXED: clean stroke state management ----
    annotations = []            # list of strokes, each stroke is a list of (x,y)
    annotationNumber = -1       # -1 means no active stroke
    annotationStart = False

    swap_handedness = False  # If your left/right still appears swapped, set True

    while True:
        success, webcam = cap.read()
        if not success:
            break
        webcam = cv2.flip(webcam, 1)

        pdf_img = images[imgNumber].copy()
        pdf_img_resized = cv2.resize(pdf_img, (width, height), interpolation=cv2.INTER_AREA)

        # ---- Compatibility across cvzone versions ----
        res = detector.findHands(webcam, flipType=True)  # usually returns (hands, img)
        if isinstance(res, tuple) and len(res) == 2:
            if isinstance(res[0], list):
                hands, img_with_hands = res
            else:
                img_with_hands, hands = res
        else:
            # fallback if API changes, assume hands list only
            hands = res
            img_with_hands = webcam

        if hands and len(hands) > 0:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            lmList = hand["lmList"]
            handType = hand["type"]

            if swap_handedness:
                handType = "Left" if handType == "Right" else "Right"

            webcam_h, webcam_w = webcam.shape[:2]
            pdf_h, pdf_w = pdf_img_resized.shape[:2]

            # map index fingertip (id 8) from webcam to PDF image and clamp
            ix = int(lmList[8][0])
            iy = int(lmList[8][1])
            mapped_x = int(ix * pdf_w / webcam_w)
            mapped_y = int(iy * pdf_h / webcam_h)
            mapped_x, mapped_y = clamp_point(mapped_x, mapped_y, pdf_w, pdf_h)
            indexFinger = (mapped_x, mapped_y)

            # Previous slide: LEFT hand with thumb up
            if fingers == [1, 0, 0, 0, 0] and handType == "Left" and not buttonPressed:
                if imgNumber > 0:
                    imgNumber -= 1
                    # reset drawing state on slide change
                    annotations = []
                    annotationNumber = -1
                    annotationStart = False
                    buttonPressed = True

            # Next slide: LEFT hand with pinky up (keep behavior as in your original)
            elif fingers == [0, 0, 0, 0, 1] and handType == "Left" and not buttonPressed:
                if imgNumber < len(images) - 1:
                    imgNumber += 1
                    # reset drawing state on slide change
                    annotations = []
                    annotationNumber = -1
                    annotationStart = False
                    buttonPressed = True

            # Pointer: index + middle up
            elif fingers == [0, 1, 1, 0, 0]:
                cv2.circle(pdf_img_resized, indexFinger, 12, (0, 0, 255), cv2.FILLED)
                annotationStart = False  # not drawing, just pointing

            # Marker: index only (draw)  ---- FIXED: safe stroke creation/indexing ----
            elif fingers == [0, 1, 0, 0, 0]:
                if not annotationStart:
                    annotationStart = True
                    annotations.append([])                  # start a new stroke
                    annotationNumber = len(annotations) - 1 # active stroke index
                if 0 <= annotationNumber < len(annotations):
                    annotations[annotationNumber].append(indexFinger)
                cv2.circle(pdf_img_resized, indexFinger, 8, (255, 0, 0), cv2.FILLED)

            # Eraser: all except thumb up  ---- FIXED: realign index after erase ----
            elif fingers == [0, 1, 1, 1, 1] and not buttonPressed:
                if annotations:
                    annotations.pop(-1)
                annotationNumber = len(annotations) - 1     # -1 if none left
                annotationStart = False
                buttonPressed = True

            else:
                annotationStart = False

        else:
            annotationStart = False

        # Debounce for slide/erase buttons
        if buttonPressed:
            buttonCounter += 1
            if buttonCounter > buttonDelay:
                buttonCounter = 0
                buttonPressed = False

        # Render all strokes
        for stroke in annotations:
            for j in range(1, len(stroke)):
                cv2.line(pdf_img_resized, stroke[j - 1], stroke[j], (255, 0, 0), 4)

        # Picture-in-picture webcam
        webcam_small = cv2.resize(webcam, (width // 4, height // 4))
        ph = height // 4
        pw = width // 4
        x1 = width - pw
        x2 = width
        y1 = 0
        y2 = ph
        pdf_img_resized[y1:y2, x1:x2] = webcam_small

        cv2.imshow("PDF with Annotations", pdf_img_resized)
        # cv2.imshow("Webcam (hands)", img_with_hands)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_presentation()
