import cv2
from ultralytics import YOLO
import time

# -------------------------
# تحميل الموديل
# -------------------------
model = YOLO("yolo26n.pt")  # غيّر الاسم حسب الموديل بتاعك

# -------------------------
# فتح كاميرا اللاب
# -------------------------
cap = cv2.VideoCapture(0)  # 0 يعني الكاميرا الافتراضية

if not cap.isOpened():
    print("مش قادر يفتح الكاميرا")
    exit()

# -------------------------
# لقياس الـ FPS
# -------------------------
prev_time = 0

# -------------------------
# حلقة عرض الفيديو وتشغيل YOLO
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("مش قادر يقرأ الإطار")
        break

    # -------------------------
    # تشغيل YOLO على الإطار
    # -------------------------
    results = model(frame)

    # -------------------------
    # رسم الصناديق على الإطار
    # -------------------------
    annotated_frame = results[0].plot()

    # -------------------------
    # حساب FPS
    # -------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # -------------------------
    # عرض الإطار
    # -------------------------
    cv2.imshow("YOLO Live", annotated_frame)

    # للخروج اضغط 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------
# إنهاء الكاميرا
# -------------------------
cap.release()
cv2.destroyAllWindows()

