import cv2
import numpy as np

img = cv2.imread("resources/RezydencjaDiabla.png", cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([img],[0],None,[256],[0,256]).ravel()
N = img.size

# Skumulowany histogram (CDF)
cdf = np.cumsum(hist)

def Hequal(g):
    return int((cdf[g] / N) * 255)

print("Hequal(40) =", Hequal(40))
print("Hequal(45) =", Hequal(45))
print("Hequal(50) =", Hequal(50))
