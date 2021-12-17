from IMPP import *

cap = cv2.VideoCapture(0)

pipeline = PostProcessingPipeline([ConvertToGray(), AverageBlur(3, True)])

cv2.namedWindow("Trackbars")

def fSizeTrack(newVal):
    if newVal < 1:
        return
    if newVal % 2 == 0:
        newVal = newVal - 1
    pipeline.blocks[1] = AverageBlur(newVal, True)

cv2.createTrackbar("Filter Size", "Trackbars", 1, 125, fSizeTrack)

while True:
    ret, frame = cap.read()
    cv2.imshow("Original", frame)

    pipeline.run(frame)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break
