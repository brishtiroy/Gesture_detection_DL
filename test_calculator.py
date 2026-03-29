import cv2
from cvzone.HandTrackingModule import HandDetector
import math
from collections import deque

def run_calculator():
    class Button:
        def __init__(self, pos, width, height, value):
            self.pos = pos; self.width = width; self.height = height; self.value = value
        def draw(self, img, is_hover=False):
            x, y = self.pos
            if is_hover: fill, border = (200, 255, 200), (0, 120, 0)
            else: fill, border = (255, 255, 255), (50, 50, 50)
            cv2.rectangle(img, (x, y), (x + self.width, y + self.height), fill, cv2.FILLED)
            cv2.rectangle(img, (x, y), (x + self.width, y + self.height), border, 3)
            cv2.putText(img, self.value, (x + 12, y + 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (50, 50, 50), 2)
        def contains(self, x, y):
            bx, by = self.pos
            return (bx < x < bx + self.width) and (by < y < by + self.height)

    def sind(x): return math.sin(math.radians(x))
    def cosd(x): return math.cos(math.radians(x))
    def tand(x): return math.tan(math.radians(x))
    def cbrt(x): return x ** (1/3)
    ALLOWED = {"sin": sind, "cos": cosd, "tan": tand, "log": math.log10, "ln": math.log, "sqrt": math.sqrt, "cbrt": cbrt, "exp": math.exp, "pi": math.pi, "e": math.e, "abs": abs, "pow": pow, "fact": math.factorial}

    def eval_expression(expr: str) -> str:
        try:
            expr = expr.replace("^", "**")
            while expr.count("(") > expr.count(")"): expr += ")"
            return f"{float(eval(expr, {'__builtins__': None}, ALLOWED)):.6g}"
        except: return "Error"

    def apply_value(eq: str, v: str) -> str:
        if v == "=": return eval_expression(eq)
        if v == "DEL": return eq[:-1]
        if v == "C": return ""
        if v == "sqrt": return eq + "sqrt("
        if v == "cbrt": return eq + "cbrt("
        if v == "x^2": return eq + "**2"
        if v == "x^3": return eq + "**3"
        if v == "pow": return eq + "**"
        if v == "pi": return eq + "pi";
        if v == "e": return eq + "e"
        if v == "exp": return eq + "exp("
        if v == "%": return eq + "/100"
        if v == "!": return f"fact({eq})" if eq else eq
        if v in ["sin", "cos", "tan", "log", "ln"]: return eq + v + "("
        return eq + v

    cap = cv2.VideoCapture(0); cap.set(3, 1600); cap.set(4, 900)
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    buttonListValues = [["sin", "cos", "tan", "log", "ln", "(", ")"], ["sqrt", "cbrt", "x^2", "x^3", "pow", "pi", "e"], ["7", "8", "9", "/", "%", "DEL", "C"], ["4", "5", "6", "*", "exp", "!", ""], ["1", "2", "3", "-", "", "", ""], ["0", ".", "=", "+", "", "", ""]]
    btn_w, btn_h, gap, start_x, start_y = 120, 90, 12, 350, 220
    buttonList = []
    for r, row in enumerate(buttonListValues):
        for c, val in enumerate(row):
            if val != "": buttonList.append(Button((start_x + c * (btn_w + gap), start_y + r * (btn_h + gap)), btn_w, btn_h, val))

    myEquation, pinch_active, hold_counter, clicked_this_pinch = "", False, 0, False
    cursor_hist = deque(maxlen=5)

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1); hands, img = detector.findHands(img, flipType=False)
        panel_x, panel_y, panel_w, panel_h = start_x, 70, 7 * btn_w + 6 * gap, 120
        cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (225, 255, 255), cv2.FILLED)
        cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (50, 50, 50), 3)
        hover_button = None

        if hands:
            # --- EXIT LOGIC ---
            fingers = detector.fingersUp(hands[0])
            if fingers[1:].count(1) == 4:
                cap.release(); cv2.destroyAllWindows(); return

            lm = hands[0]["lmList"]
            cursor_hist.append(lm[8][:2])
            sx, sy = int(sum(p[0] for p in cursor_hist)/len(cursor_hist)), int(sum(p[1] for p in cursor_hist)/len(cursor_hist))
            dist, _, img = detector.findDistance(lm[8][:2], lm[12][:2], img)
            for b in buttonList:
                if b.contains(sx, sy): hover_button = b; break
            cv2.circle(img, (sx, sy), 10, (255, 0, 255), cv2.FILLED)
            if not pinch_active:
                if dist < 45: pinch_active, hold_counter, clicked_this_pinch = True, 1, False
            else:
                if dist < 45: hold_counter += 1
                elif dist > 65: pinch_active, clicked_this_pinch = False, False
            if pinch_active and hold_counter >= 3 and not clicked_this_pinch and hover_button:
                myEquation = apply_value(myEquation, hover_button.value); clicked_this_pinch = True

        for b in buttonList: b.draw(img, is_hover=(hover_button == b))
        cv2.putText(img, myEquation, (panel_x + 15, panel_y + 80), cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 50), 3)
        cv2.imshow("Scientific Hand Calculator", img)
        if cv2.waitKey(1) == ord("q"): break
    cap.release(); cv2.destroyAllWindows()