import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# --- IMPORT YOUR MODULES ---
try:
    from blackboard import run_blackboard
    from presentation import run_presentation
    from test_calculator import run_calculator
except ImportError as e:
    print(f"Error importing modules: {e}")


def run_main_menu():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    counter = 0
    selection_threshold = 25
    current_selection = -1

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)

        # UI Drawing
        cv2.putText(img, "VISION MENU", (500, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img, "1 Finger -> Blackboard", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, "2 Fingers -> Presentation", (70, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(img, "3 Fingers -> Calculator", (70, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        cv2.putText(img, "4 Fingers -> EXIT", (70, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if hands:
            # Get the list of fingers [Thumb, Index, Middle, Ring, Pinky]
            fingers = detector.fingersUp(hands[0])

            # IMPROVED LOGIC: Count only the non-thumb fingers for menu selection
            # This prevents the thumb from accidentally triggering the next option
            actual_count = fingers[1:].count(1)

            if actual_count == 4:
                selection = 4
            elif actual_count == 3:
                selection = 3
            elif actual_count == 2:
                selection = 2
            elif actual_count == 1:
                selection = 1
            else:
                selection = -1
                counter = 0

            if selection != -1:
                if selection == current_selection:
                    counter += 1
                else:
                    current_selection = selection
                    counter = 0
                cv2.circle(img, (1100, 300), counter * 4, (0, 255, 0), 5)

            if counter > selection_threshold:
                if current_selection == 4: break

                cap.release()
                cv2.destroyAllWindows()

                if current_selection == 1:
                    run_blackboard()
                elif current_selection == 2:
                    run_presentation()
                elif current_selection == 3:
                    run_calculator()

                run_main_menu()  # Re-open menu after module closes
                return

        cv2.imshow("Main Vision Menu", img)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_main_menu()