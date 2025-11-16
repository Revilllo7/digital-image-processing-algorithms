import cv2
import numpy as np

img = cv2.imread("resources/RezydencjaDiabla.png", cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([img],[0],None,[256],[0,256]).ravel()
N = img.size

# Skumulowany histogram (CDF)
cdf = np.cumsum(hist)

alpha = -1/3

def Hhyper(g, alpha=alpha):
    return int(255 * ( (g/255)**alpha ))

print("Hhyper(30) =", Hhyper(30))
print("Hhyper(45) =", Hhyper(45))
print("Hhyper(50) =", Hhyper(50))