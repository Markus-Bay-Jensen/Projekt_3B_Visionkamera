import cv2
import numpy as np
from matplotlib import pyplot as plt

from IMPP import *

cap = cv2.VideoCapture(0)

pipeline = PostProcessingPipeline([
    #GetRedChannel(),
    ConvertToGray(showOutput = True, outputWindowName = 'test window'),
    #AverageBlur(7, True),
    #GaussianBlur(201),
    #BilateralFilter(5, 10, 10, True)
    #LaplacianSharpen(0.5, True, True)
    #UnsharpMasking(7, True)
    #Threshold(200, 250, showOutput = True),
    #AdaptiveThreshold(200, 11, 2, showOutput = True),
    #OtsuBinarization(True)
    #DetectContours(draw = True, drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    #ThresholdContours(100, 8000, printDebug = True),
    #DetectShapes(printResult = True)
])

while(True):
    ret, frame = cap.read()
    pipeRes = pipeline.run(frame)

    img = pipeRes
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    #plt.show()
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    # create a mask first, center square is 1, remaining all zeros
    #mask = np.zeros((rows,cols,2),np.uint8)
    #mask[crow-3:crow+3, ccol-3:ccol+3] = 1
    # apply mask and inverse DFT
    fshift = dft_shift #*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    #plt.show()

    cv2.imshow("Webcam", frame)
    cv2.imshow("hz",img_back)
    #cv2.imshow("Shapes", shapeImg)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
