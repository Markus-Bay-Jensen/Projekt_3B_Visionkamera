from pipeline import *
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    Firkan_Liste = PipeRes(frame.copy())
    Firkan_2 = []
    Firkan_Liste = PipeRes(frame.copy())
    frame_Firkan =frame.copy()
    for f in Firkan_Liste:
        #print('Firkan_Liste',f)
        print(' P:',f[1].points,' F:',f[0])
        if f[1].points == 4:
            Firkan_2.append([f[1].approx,f[0]])
            print('P:',f[1].points,' F:',f[0])
        #cv2.drawContours(frame_Firkan, [f.contour], -1, (0, 255, 0), 2)
       
    cv2.imshow("Firkan", frame_Firkan)
    if cv2.waitKey(100) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
