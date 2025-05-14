import numpy as np
import cv2
import time
from pyzbar.pyzbar import decode

# Detected barcodes dictionary to avoid repeat detections
detected_barcodes = {}

def preprocess(frame):
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    grad_X = cv2.Sobel(gray_scale, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_Y = cv2.Sobel(gray_scale, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(grad_X, grad_Y)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.blur(gradient, (7, 7))
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    return closed

def decode_barcodes(frame):
    global detected_barcodes
    barcodes = decode(frame)
    current_time = time.time()

    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        
        if barcode_data not in detected_barcodes or current_time - detected_barcodes[barcode_data] > 10:
            detected_barcodes[barcode_data] = current_time
            print("Barcode Founded. Type: {} Data: {}".format(barcode_type, barcode_data))

        
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        text = "Barcode data :{} Barcode Type: {}".format(barcode_data, barcode_type)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

def detect_contours_and_draw(frame):
    closed = preprocess(frame)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        rct = cv2.minAreaRect(c)
        box_contour = np.int32(cv2.boxPoints(rct))
        cv2.drawContours(frame, [box_contour], -1, (0, 255, 0), 3)

    return frame


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Couldnt open webcam.")
    exit()

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    decoded_frame = decode_barcodes(gray_frame)

    processed_frame = detect_contours_and_draw(frame)

    cv2.imshow("Barcode Detection (Grayscale)", decoded_frame)
    cv2.imshow("Contour Detection (Processed)", processed_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
