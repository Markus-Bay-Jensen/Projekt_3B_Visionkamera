import AI
import cv2

cap = cv2.VideoCapture(0)
m2 = AI.Data('RGB')
ai2 = AI.AI('RGB',1)
file = m2.LoadData()
while True:
    while True:
        ret, frame = cap.read()

        frame2 = frame.copy()
        m3,ai = ai2.ai(frame[300][200])
        cv2.putText(frame2, str(ai), [300,220], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
        cv2.putText(frame2, str('X'), [298,198], cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
        cv2.imshow("Webcam", frame2)
        m = cv2.waitKey(1)
        if m == ord('q'):
            m2.SaveData(file)
            break
        if m != -1:
            break
    if m == ord('q'):
        break    
        
    RGB = frame[300][200]
    print(m3)
    print(RGB)
    R = RGB[0]
    G = RGB[1]
    B = RGB[2]
    userIn = input("frame: ")
    if userIn != '':
        RGB = [userIn,[R,G,B]]
        file.append(RGB)

cap.release()
cv2.destroyAllWindows()