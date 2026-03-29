import cv2
import os
import numpy as np
import fitz
from cvzone.HandTrackingModule import HandDetector


def pdf_to_images(path):
    doc = fitz.open(path);
    imgs = []
    for i in range(len(doc)):
        page = doc.load_page(i);
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        imgs.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    doc.close();
    return imgs


def clamp_point(x, y, w, h): return max(0, min(x, w - 1)), max(0, min(y, h - 1))


def run_presentation():
    width, height = 1400, 800
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file.pdf")
    images = pdf_to_images(pdf_path)
    if not images: return
    cap = cv2.VideoCapture(0);
    cap.set(3, width);
    cap.set(4, height)
    detector = HandDetector(detectionCon=0.75, maxHands=1)
    imgNumber, buttonPressed, buttonCounter, buttonDelay = 0, False, 0, 20
    annotations, annotationNumber, annotationStart = [], -1, False
    swap_handedness = False

    while True:
        success, webcam = cap.read()
        if not success: break
        webcam = cv2.flip(webcam, 1)
        pdf_img_resized = cv2.resize(images[imgNumber].copy(), (width, height), interpolation=cv2.INTER_AREA)
        res = detector.findHands(webcam, flipType=True)
        if isinstance(res, tuple) and len(res) == 2:
            hands = res[0]
        else:
            hands = res

        if hands:
            hand = hands[0];
            fingers = detector.fingersUp(hand)
            lmList = hand["lmList"];
            handType = hand["type"]
            if swap_handedness: handType = "Left" if handType == "Right" else "Right"

            # --- EXIT LOGIC ---
            if fingers[1:].count(1) == 4:
                cap.release();
                cv2.destroyAllWindows();
                return

            webcam_h, webcam_w = webcam.shape[:2];
            pdf_h, pdf_w = pdf_img_resized.shape[:2]
            mapped_x, mapped_y = clamp_point(int(lmList[8][0] * pdf_w / webcam_w), int(lmList[8][1] * pdf_h / webcam_h),
                                             pdf_w, pdf_h)
            indexFinger = (mapped_x, mapped_y)

            if fingers == [1, 0, 0, 0, 0] and handType == "Left" and not buttonPressed:
                if imgNumber > 0: imgNumber -= 1; annotations, annotationNumber, annotationStart, buttonPressed = [], -1, False, True
            elif fingers == [0, 0, 0, 0, 1] and handType == "Left" and not buttonPressed:
                if imgNumber < len(
                    images) - 1: imgNumber += 1; annotations, annotationNumber, annotationStart, buttonPressed = [], -1, False, True
            elif fingers == [0, 1, 1, 0, 0]:
                cv2.circle(pdf_img_resized, indexFinger, 12, (0, 0, 255), cv2.FILLED);
                annotationStart = False
            elif fingers == [0, 1, 0, 0, 0]:
                if not annotationStart: annotationStart = True; annotations.append([]); annotationNumber = len(
                    annotations) - 1
                if 0 <= annotationNumber < len(annotations): annotations[annotationNumber].append(indexFinger)
                cv2.circle(pdf_img_resized, indexFinger, 8, (255, 0, 0), cv2.FILLED)
            elif fingers == [0, 1, 1, 1, 1] and not buttonPressed:
                if annotations: annotations.pop(-1)
                annotationNumber = len(annotations) - 1;
                annotationStart, buttonPressed = False, True
            else:
                annotationStart = False
        else:
            annotationStart = False

        if buttonPressed:
            buttonCounter += 1
            if buttonCounter > buttonDelay: buttonCounter, buttonPressed = 0, False
        for stroke in annotations:
            for j in range(1, len(stroke)): cv2.line(pdf_img_resized, stroke[j - 1], stroke[j], (255, 0, 0), 4)
        webcam_small = cv2.resize(webcam, (width // 4, height // 4))
        pdf_img_resized[0:height // 4, width - width // 4:width] = webcam_small
        cv2.imshow("PDF with Annotations", pdf_img_resized)
        if cv2.waitKey(1) == 27: break
    cap.release();
    cv2.destroyAllWindows()