import cv2
import os

cap = cv2.VideoCapture(0)

label = input("Enter gesture name (one/two/three): ")
save_path = f"dataset/{label}"
os.makedirs(save_path, exist_ok=True)

count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        img = cv2.resize(frame, (64,64))
        cv2.imwrite(f"{save_path}/{count}.jpg", img)
        count += 1
        print("Saved:", count)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()