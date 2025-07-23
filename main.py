import cv2
import requests
import winsound  
from ultralytics import YOLO
import os
from datetime import datetime
import time

BOT_TOKEN = '8037950501:AAFZGTIhG6WKafogoblSvMOUYSFn0zACBJU'
CHAT_ID = '1282149880'
CONFIDENCE_THRESHOLD = 0.3
TELEGRAM_INTERVAL = 30  
MODEL_PATH = 'yolov8n.pt' 


model = YOLO(MODEL_PATH)
last_alert_time = 0  

def send_telegram_image(image_path):
    print("[INFO] Sending image via Telegram...")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    
    with open(image_path, 'rb') as photo:
        response = requests.post(
            url,
            data={'chat_id': CHAT_ID, 'caption': '📱 ALERT: Mobile Phone Detected!'},
            files={'photo': photo}
        )

    if response.status_code == 200:
        print("[✅] Telegram alert sent successfully!")
    else:
        print("[❌] Telegram error:", response.text)
        
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read from camera.")
        break

    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, save=False, verbose=False)

    detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            if class_name == 'cell phone':
                detected = True
                print(f"[DETECTED] Cell phone ({conf:.2f})")
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    current_time = time.time()
    if detected and (current_time - last_alert_time) > TELEGRAM_INTERVAL:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"detected_{timestamp}.jpg"
        cv2.imwrite(filepath, frame)
        
        winsound.Beep(1500, 300)  

        send_telegram_image(filepath)
        os.remove(filepath)  
        last_alert_time = current_time
        
    cv2.imshow("📹 CCTV Mobile Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
