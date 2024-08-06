# implement derivative based edge detection
import numpy as np
import cv2
import matplotlib.pyplot as plt
from machine import Pin, ADC
import time

def edge_detection(img):
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply sobel filter
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    # calculate magnitude and direction
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)
    return magnitude, direction
def main():
    img = cv2.imread('edge.jpg')
    magnitude, direction = edge_detection(img)
    plt.figure()
    plt.imshow(magnitude, cmap='gray')
    plt.title('Magnitude of Derivative')
    plt.show()
if __name__ == '__main__':
    main()



gs0 = ADC(Pin(26))
gs1 = ADC(Pin(27))
gs2 = ADC(Pin(28))
edge_ref = 1000    # edge_reference
line_ref = 10000 # line_reference

def get_value():
    return [gs0.read_u16(), gs1.read_u16(), gs2.read_u16()]

def is_on_edge():
    gs_list = get_value()
    for value in gs_list:
        if value<=edge_ref:
            return True
    return False

def get_line_status():
    gs_list = get_value()
    line_status=[]
    for value in gs_list:
        line_status.append(value<line_ref)
    return line_status

while True:
    print(get_value())
    time.sleep(0.2)
    if is_on_edge():
        print("Danger!")
    else:
        print(get_line_status())