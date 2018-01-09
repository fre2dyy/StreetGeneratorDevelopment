# coding=utf-8

import sys
sys.path
import time
import matplotlib
#import matplotlib.pyplot
import matplotlib.colors
import operator
# from osgeo import gdal, gdalnumeric, ogr, osr
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import PIL.ImageOps
Image.MAX_IMAGE_PIXELS = 1000000000
from PIL import ImageDraw
import numpy as np
import cv2


Ilmenau_streets = Image.open("files/shadow/streets_shadow_EPSG.tiff")
# Einlesen der Bilder


################################################################################################
############### ABLAUF DIESES ALGORITHMUS ######################################################
#
# 1. Kontrastanpassungen
# 2. Manipulationen im HSV-Farbraum (Selektion: hoher Hellwert & geringer Sättigungswert)
# -> Ausgabe: Binärbild (schwarz: Straßen & Umgebung, weiß: Farbbahnmarkierung und "Störpixel"
#
################################################################################################


def imagetonumpy(picture):             # Umwandlung der Bilder in Arrays
    np_img = np.asarray(picture)                 # n_img = Numpy-Array des eingelesenen Bildes
    np_img = np_img.astype(float)
    np_img_noa = np_img[:, :, 0:3]                # Entfernung des Alpha-Layers
    np_img_noa = np_img_noa / 255
    return np_img_noa


def numpytoimage(numpy):                # Umwandlung der Arrays in Bilder
    numpy = numpy * 255
    n_img_noa_original = Image.fromarray(numpy.astype(np.uint8))        # TEST: Rückumwandlung von Array in Bild
    # n_img_noa_original_inv = PIL.ImageOps.invert(n_img_noa_original)    # Erzeugen des Negativ-Bildes
    # n_img_noa_original_inv.show()
    return n_img_noa_original

def change_contrast(img, level):
    #img = Image.open(img)
    img.load()

    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        return 128 + factor * (c - 128)

    return img.point(contrast)


Ilmenau_streets = change_contrast(Ilmenau_streets, 150)    # Kontrast erhöhen
# Ilmenau_streets.save("files/shadow/streets_shadow.tiff")


Ilmenau_streets.show()

Ilmenau_streets2_numpy = imagetonumpy(Ilmenau_streets)
#Ilmenau_streets2_numpy[:,:,:].tofile('C:\Users\student\PyCharm\Masterarbeit\output\Test\Templates\orig\Foooooo.csv',sep=',',format='%10.2f')



#######################################################################
# optional
#######################################################################

i = 0
j = 0

while i < Ilmenau_streets2_numpy.shape[0]:
    while j < Ilmenau_streets2_numpy.shape[1]:
         if Ilmenau_streets2_numpy[i, j, 0] +  Ilmenau_streets2_numpy[i, j, 1] + Ilmenau_streets2_numpy[i, j, 2] < 1:

             Ilmenau_streets2_numpy[i, j, :] = 0


         else:
             Ilmenau_streets2_numpy[i, j, :] = Ilmenau_streets2_numpy[i, j, :]

         j = j + 1

    j = 0
    i= i + 1
    print(i)


Ilmenau_streets2_hsv = matplotlib.colors.rgb_to_hsv(Ilmenau_streets2_numpy)

#######################################################################

i = 0
j = 0

while i < Ilmenau_streets2_hsv.shape[0]:
    while j < Ilmenau_streets2_hsv.shape[1]:
         if Ilmenau_streets2_hsv[i, j, 1] > 0.4:
             Ilmenau_streets2_hsv[i, j, :] = 0
         else:
             Ilmenau_streets2_hsv[i, j, :] = Ilmenau_streets2_hsv[i, j, :]


         if Ilmenau_streets2_hsv[i, j, 2] < 0.65:
             Ilmenau_streets2_hsv[i, j, :] = 0
         else:
             Ilmenau_streets2_hsv[i, j, 0] = 0
             Ilmenau_streets2_hsv[i, j, 1] = 0
             Ilmenau_streets2_hsv[i, j, 2] = 1

         j = j + 1

    j = 0
    i= i + 1
    print(i)


Ilmenau_streets2_rgb = matplotlib.colors.hsv_to_rgb(Ilmenau_streets2_hsv)  # Umwandlung HSV in RGB
Ilmenau_streets3 = numpytoimage(Ilmenau_streets2_rgb)


Ilmenau_streets3.show()
##### Bildspeicherung #####
path = 'files/shadow/'
file = 'streets_shadow_hsv'    # + '(' + time.strftime("%d-%m-%y") + ')'
ending = '.tiff'
complete_path = path + file + ending

print(complete_path)
Ilmenau_streets3.save(complete_path)
