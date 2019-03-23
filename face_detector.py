import cv2

smile_detector = cv2.CascadeClassifier("./smile.xml")
eye_detector = cv2.CascadeClassifier("./eyes.xml")

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smile = smile_detector.detectMultiScale(frame, scaleFactor=2.0, minNeighbors=20)

    for (x, y, w, h) in smile:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

capture.release()
cv2.destroyAllWindows()
