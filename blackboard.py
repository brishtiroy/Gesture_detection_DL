import mediapipe as mp
import cv2
import numpy as np
import time
import os


def run_blackboard():
    ml = 150
    max_x, max_y = 250 + ml, 50
    curr_tool = "select tool"
    time_init = True
    rad = 40
    var_inits = False
    thick = 4
    prevx, prevy = 0, 0
    selected_color = (255, 255, 255)

    # --- Smoothing Variables ---
    smooth_x, smooth_y = 0, 0
    alpha = 0.40  # Lower = More Latency/Smoothness, Higher = Snappier/Jittery

    def getTool(x):
        if x < 50 + ml:
            return "line"
        elif x < 100 + ml:
            return "rectangle"
        elif x < 150 + ml:
            return "draw"
        elif x < 200 + ml:
            return "circle"
        else:
            return "erase"

    def index_raised(yi, y9):
        return (y9 - yi) > 40

    def select_color_smooth(x):
        nonlocal selected_color
        if x < 50:
            target_color = (255, 0, 0)
        elif x < 100:
            target_color = (0, 255, 0)
        elif x < 150:
            target_color = (0, 0, 255)
        elif x < 200:
            target_color = (0, 255, 255)
        else:
            target_color = (255, 255, 255)
        selected_color = tuple(int(a * 0.8 + b * 0.2) for a, b in zip(selected_color, target_color))

    hands = mp.solutions.hands
    hand_landmark = hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.8, max_num_hands=1)
    draw = mp.solutions.drawing_utils

    blackboard = np.zeros((600, 800, 3), dtype=np.uint8)

    palette = np.zeros((300, 50, 3), dtype=np.uint8)
    palette[0:50, :] = [255, 0, 0]
    palette[50:100, :] = [0, 255, 0]
    palette[100:150, :] = [0, 0, 255]
    palette[150:200, :] = [0, 255, 255]
    palette[200:250, :] = [255, 255, 255]

    clear_button_size = (100, 40)
    clear_button_pos = (590, 0)
    save_button_size = (100, 40)
    save_button_pos = (700, 0)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    try:
        tools = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pic1.jpg'))
        tools = cv2.resize(tools, (max_x - ml, max_y))
    except:
        tools = np.zeros((max_y, max_x - ml, 3), dtype=np.uint8)  # Fallback if image missing

    while True:
        _, frm = cap.read()
        frm = cv2.flip(frm, 1)
        frm = cv2.resize(frm, (800, 600))
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        op = hand_landmark.process(rgb)

        blackboard_copy = blackboard.copy()
        x, y = 0, 0  # Default values if no hand is detected

        if op.multi_hand_landmarks:
            for i in op.multi_hand_landmarks:
                draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)

                # Raw coordinates
                raw_x = int(i.landmark[8].x * 800)
                raw_y = int(i.landmark[8].y * 600)

                # Applying Exponential Smoothing (LPE Filter)
                smooth_x = int(smooth_x * (1 - alpha) + raw_x * alpha)
                smooth_y = int(smooth_y * (1 - alpha) + raw_y * alpha)

                # Use smoothed coordinates for everything below
                x, y = smooth_x, smooth_y

                # Tool selection logic
                if ml < x < max_x and y < max_y:
                    if time_init:
                        ctime = time.time()
                        time_init = False
                    ptime = time.time()
                    cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                    rad -= 1
                    if (ptime - ctime) > 0.8:
                        curr_tool = getTool(x)
                        time_init = True
                        rad = 40
                else:
                    time_init = True
                    rad = 40

                if 0 < x < 50:
                    select_color_smooth(y)

                # Drawing logic
                if curr_tool == "draw":
                    yi = int(i.landmark[12].y * 600)
                    y9 = int(i.landmark[9].y * 600)
                    if index_raised(yi, y9):
                        cv2.line(blackboard, (prevx, prevy), (x, y), selected_color, thick)
                    prevx, prevy = x, y

                elif curr_tool in ["line", "rectangle", "circle"]:
                    yi = int(i.landmark[12].y * 600)
                    y9 = int(i.landmark[9].y * 600)
                    if index_raised(yi, y9):
                        if not var_inits:
                            xii, yii = x, y
                            var_inits = True
                    else:
                        if var_inits:
                            if curr_tool == "line":
                                cv2.line(blackboard, (xii, yii), (x, y), selected_color, thick)
                            elif curr_tool == "rectangle":
                                cv2.rectangle(blackboard, (xii, yii), (x, y), selected_color, thick)
                            elif curr_tool == "circle":
                                radius = int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                                cv2.circle(blackboard, (xii, yii), radius, selected_color, thick)
                            var_inits = False

                elif curr_tool == "erase":
                    yi = int(i.landmark[12].y * 600)
                    y9 = int(i.landmark[9].y * 600)
                    if index_raised(yi, y9):
                        cv2.circle(blackboard, (x, y), 30, (0, 0, 0), -1)

        # UI Overlay Logic
        cv2.circle(blackboard_copy, (x, y), 5, (0, 0, 255), -1)
        blackboard_copy[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, blackboard_copy[:max_y, ml:max_x], 0.3, 0)
        blackboard_copy[0:300, 0:50] = palette

        # Buttons
        cv2.rectangle(blackboard_copy, clear_button_pos,
                      (clear_button_pos[0] + clear_button_size[0], clear_button_pos[1] + clear_button_size[1]),
                      (255, 255, 255), -1)
        cv2.putText(blackboard_copy, "Clear", (clear_button_pos[0] + 10, clear_button_pos[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(blackboard_copy, save_button_pos,
                      (save_button_pos[0] + save_button_size[0], save_button_pos[1] + save_button_size[1]),
                      (255, 255, 255), -1)
        cv2.putText(blackboard_copy, "Save", (save_button_pos[0] + 10, save_button_pos[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(blackboard_copy, "Current Tool: " + curr_tool.capitalize(), (10, 580), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.imshow("Virtual Blackboard", blackboard_copy)

        # Interaction with buttons
        if clear_button_pos[0] < x < clear_button_pos[0] + clear_button_size[0] and clear_button_pos[1] < y < \
                clear_button_pos[1] + clear_button_size[1]:
            blackboard = np.zeros((600, 800, 3), dtype=np.uint8)

        if save_button_pos[0] < x < save_button_pos[0] + save_button_size[0] and save_button_pos[1] < y < \
                save_button_pos[1] + save_button_size[1]:
            # Simple Save Implementation
            cv2.imwrite("saved_drawing.png", blackboard)
            print("Saved as saved_drawing.png")
            time.sleep(0.5)  # Prevent multiple saves

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_blackboard()