import cv2

from test_calculator import run_calculator
from blackboard import run_blackboard
from presentation import run_presentation

while True:
    # Create simple menu screen
    img = cv2.imread("black.jpg") if False else None
    frame = 255 * (cv2.UMat(600, 800, cv2.CV_8UC3).get())  # white screen

    cv2.putText(frame, "MAIN MENU", (250, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    cv2.putText(frame, "Press 1: Blackboard", (200, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.putText(frame, "Press 2: Presentation", (200, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.putText(frame, "Press 3: Calculator", (200, 390),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.putText(frame, "ESC to Exit", (200, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Menu", frame)

    key = cv2.waitKey(1)

    if key == ord('1'):
        cv2.destroyAllWindows()
        run_blackboard()

    elif key == ord('2'):
        cv2.destroyAllWindows()
        run_presentation()

    elif key == ord('3'):
        cv2.destroyAllWindows()
        run_calculator()

    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()