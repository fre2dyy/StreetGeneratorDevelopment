# coding=utf-8


import sys
sys.path
import time
import matplotlib
#import matplotlib.pyplot
import matplotlib.colors
import matplotlib.image
import operator
import gdal
#from osgeo import gdal, gdalnumeric, ogr, osr
from PIL import Image
from PIL import ImageEnhance
import PIL.ImageOps
Image.MAX_IMAGE_PIXELS = 1000000000
from PIL import ImageDraw
import numpy as np
import cv2


Ilmenau_streets2 = Image.open("C:\Users\student\QGIS\output\streets_centre_imp.tif")                    # Einlesen der Bilder
Ilmenau_streets = Image.open("files/colour/streets_clipped.tiff")

################################################################################################
############### ABLAUF DIESES ALGORITHMUS ######################################################
#
# 1. Transformation der Bilder in numpy-Arrays + Entfernung von möglichen Alpha Layern
# 2. Schattenkorrektur
# 2.1. Extraktion der Schatten
# 2.2. Beschneidung der Ränder der extrahierten Schatten (SHADOW MASK)
# 2.3. Kontrast- und Farbanpassungen der Schattenbereiche an das restliche Bild
# 2.4. Mergen von "Schatten" mit dem restlichen Bild
#
################################################################################################


#Ilmenau_streets_gdal = gdal.Open(Ilmenau_streets)


# n_img = Numpy-Array des eingelesenen Bildes
# n_img_noa = n_img ohne Alpha-Layer
# n_img_hsv = n_img_noa im HSV-Farbraum
#
#
"""
--------------------------------------------------------------
FUNKTIONEN
--------------------------------------------------------------
"""
def imagetonumpy(picture):             # Umwandlung der Bilder in Arrays
    np_img = np.asarray(picture)                 # n_img = Numpy-Array des eingelesenen Bildes
    np_img = np_img.astype(float)
    np_img_noa = np_img[:, :, 0:3]                # Entfernung des Alpha-Layers
    #np_img_noa = np_img_noa[...,::-1]
    np_img_noa = np_img_noa / 255
    return np_img_noa


def numpytoimage(numpy):                # Umwandlung der Arrays in Bilder
    numpy = numpy * 255
    n_img_noa_original = Image.fromarray(numpy.astype(np.uint8))        # TEST: Rückumwandlung von Array in Bild
    n_img_noa_original_inv = PIL.ImageOps.invert(n_img_noa_original)    # Erzeugen des Negativ-Bildes
    # n_img_noa_original_inv.show()
    return n_img_noa_original_inv


def shadowmask(numpy):          # Schatten Maske
    mask = numpy

    i = 0
    j = 0

    while i < mask.shape[0]:
        while j < mask.shape[1]:
            if mask[i, j, 0] == 1 and mask[i, j, 1] == 1 and mask[i, j, 2] == 1:
                mask[i, j, 0] = 0
                mask[i, j, 1] = 0
                mask[i, j, 2] = 1
            else:
                mask[i, j, 0] = 0
                mask[i, j, 1] = 0
                mask[i, j, 2] = 0

            j = j + 1
        j = 0
        i = i + 1

    return mask


"""
--------------------------------------------------------------------------------------------------------------------------
Transformation der Bilder in numpy-Arrays + Entfernung von möglichen Alpha Layern
--------------------------------------------------------------------------------------------------------------------------
"""

# n_img_noa = imagetonumpy(Ilmenau_streets)
# n_img_noa_original_inv = numpytoimage(n_img_noa)
#
# path = 'C:\Users\student\PyCharm\Masterarbeit\output'
# file = '\Original' + '(' + time.strftime("%d-%m-%y") + ')'
# ending = '.tif'
# complete_path = path + file + ending
#
# print(complete_path)
# n_img_noa_original_inv.save(complete_path)



"""
--------------------------------------------------------------
Schattenextraktion
--------------------------------------------------------------
"""

##### Verarbeitung Rohbild #####
n_img_noa = imagetonumpy(Ilmenau_streets)
print n_img_noa.shape
Ilmenau_streets_shadow = numpytoimage(n_img_noa)
Ilmenau_streets_noshadow = PIL.ImageOps.invert(Ilmenau_streets_shadow)

n_img_noa = imagetonumpy(Ilmenau_streets_shadow)                    # Extrahierte Schattenbereiche (Umwandlung Bild in Numpy)
n_img_hsv_shadow = matplotlib.colors.rgb_to_hsv(n_img_noa)           # Extrahierte Schattenbereiche (Umwandlung RGB in HSV)


i = 0
j = 0

while i < n_img_hsv_shadow.shape[0]:       # Extraktion der Schatten (übrig bleiben die Schattenbereiche)
    while j < n_img_hsv_shadow.shape[1]:
         if n_img_hsv_shadow[i, j, 2] >= 0.75 and n_img_hsv_shadow[i, j, 2] < 1:
             n_img_hsv_shadow[i, j, 2] = n_img_hsv_shadow[i, j, 2]
         else:
             n_img_hsv_shadow[i, j, :] = 1

         j = j + 1
    j = 0
    i= i + 1



####################################################################################################################################################
# SHADOW MASK
####################################################################################################################################################

n_img_rgb_shadow = matplotlib.colors.hsv_to_rgb(n_img_hsv_shadow)  # Schatten-Array in Schatten-Bild (Zwischenschritt nötig)
real_img_shadow = numpytoimage(n_img_rgb_shadow)
real_img_shadow = PIL.ImageOps.invert(real_img_shadow)
#real_img_shadow.show()

shadow_mask_hsv = shadowmask(n_img_hsv_shadow)      # Erstellung der Schatten MASKE

shadow = imagetonumpy(real_img_shadow)  # Schatten-Bild in Schatten-Array (Zwischenschritt nötig)
shadow_hsv = matplotlib.colors.rgb_to_hsv(shadow)


############ Speicherung der Schattenmaske ############
#######################################################

n_img_rgb_shadow = matplotlib.colors.hsv_to_rgb(shadow_hsv)  # Umwandlung HSV in RGB
real_img_shadow = numpytoimage(n_img_rgb_shadow)
#real_img_shadow = PIL.ImageOps.invert(real_img_shadow)

#real_img_shadow.show()


n_img_rgb_shadow = matplotlib.colors.hsv_to_rgb(shadow_mask_hsv)  # Umwandlung HSV in RGB
shadow_mask_hsv_pict = numpytoimage(n_img_rgb_shadow)
#shadow_mask_hsv_pict = PIL.ImageOps.invert(shadow_mask_hsv_pict)

#shadow_mask_hsv_pict.show()


path = 'C:\Users\student\PyCharm\Masterarbeit\output'
file = '\Shadows_trash' + '(' + time.strftime("%d-%m-%y") + ')'
ending = '.png'
complete_path_trash = path + file + ending

shadow_mask_hsv_pict.save(complete_path_trash)


############ Einlesen der Schattenmaske ############
#######################################################

shadow_mask_hsv_pict = cv2.imread(complete_path_trash)
imgray_a = cv2.cvtColor(shadow_mask_hsv_pict,cv2.COLOR_BGR2GRAY)
ret_a,thresh_a = cv2.threshold(imgray_a,127,255,0)
print thresh_a
print np.shape(thresh_a)


#===================
# FIND CONTOURS
#===================
_, ca, _ = cv2.findContours(thresh_a, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print "shape: ", np.shape(ca[0])


#===================
# DRAW CONTOURS
#===================
cv2.drawContours(shadow_mask_hsv_pict, ca, -1, (0,0,0), 1)

# cv2.imshow('2', shadow_mask_hsv_pict)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#################################################################


i = 0
j = 0

while i < shadow_hsv.shape[0]:
    while j < shadow_hsv.shape[1]:
         if shadow_mask_hsv_pict[i, j, 0] == 0 and shadow_mask_hsv_pict[i, j, 1] == 0 and shadow_mask_hsv_pict[i, j, 2] == 0:
             shadow_hsv[i, j, :] = 1
         else:
             shadow_hsv[i, j, :] = shadow_hsv[i, j, :]

         j = j + 1
    j = 0
    i= i + 1

n_img_rgb_shadow = matplotlib.colors.hsv_to_rgb(shadow_hsv)  # Umwandlung HSV in RGB
real_img_shadow = numpytoimage(n_img_rgb_shadow)

#real_img_shadow.show


############ Speicherung der finalen Schattenmaske ############
###############################################################
# zu helle Ränder der Schattenbereiche wurden beschnitten
# Fehldeuten von hellen Rändern und Straßenmarkierung behoben

path = 'C:\Users\student\PyCharm\Masterarbeit\output'
file = '\Shadows_MASK_FINAL' + '(' + time.strftime("%d-%m-%y") + ')'
ending = '.png'
complete_path = path + file + ending

print(complete_path)
real_img_shadow.save(complete_path)



####################################################################################################################################################
####################################################################################################################################################



# n_img_rgb_shadow = matplotlib.colors.hsv_to_rgb(n_img_hsv_shadow)  # Umwandlung HSV in RGB
# real_img_shadow = numpytoimage(n_img_rgb_shadow)


##### Bildbearbeitungen #####
real_img_shadow = ImageEnhance.Brightness(real_img_shadow).enhance(3.0)
real_img_shadow = ImageEnhance.Color(real_img_shadow).enhance(0.1)

#real_img_shadow.show()


##### Bildspeicherung #####
path = 'C:\Users\student\PyCharm\Masterarbeit\output'
file = '\Shadows' + '(' + time.strftime("%d-%m-%y") + ')'
ending = '.png'
complete_path = path + file + ending

print(complete_path)
real_img_shadow.save(complete_path)



##### Kombination aus Originalbild (Ilmenau_streets_noshadow) und aufgehellten Schatten (real_img_shadow) #####

#real_img_shadow.show()
#Ilmenau_streets_noshadow.show()

n_img_noa = imagetonumpy(real_img_shadow)               # Schatten
n_img_noa2 = imagetonumpy(Ilmenau_streets_noshadow)     # Originalbild

n_img_hsv_shadow = matplotlib.colors.rgb_to_hsv(n_img_noa)        # Umwandlung RGB in HSV
n_img_hsv_shadow2 = matplotlib.colors.rgb_to_hsv(n_img_noa2)      # Umwandlung RGB in HSV

i = 0
j = 0

# while i < n_img_hsv_shadow.shape[0]:
#     while j < n_img_hsv_shadow.shape[1]:
#          if n_img_hsv_shadow[i, j, 1] == 0.38:
#              n_img_hsv_shadow2[i, j, :] = n_img_hsv_shadow2[i, j, :]
#          else:
#              n_img_hsv_shadow2[i, j, :] = n_img_hsv_shadow[i, j, :]
#
#          j = j + 1
#     j = 0
#     i= i + 1

while i < n_img_hsv_shadow.shape[0]:
    while j < n_img_hsv_shadow.shape[1]:
         if n_img_hsv_shadow[i, j, 1] > 0.13:
             n_img_hsv_shadow2[i, j, :] = n_img_hsv_shadow2[i, j, :]
         else:
             n_img_hsv_shadow2[i, j, :] = n_img_hsv_shadow[i, j, :]

         j = j + 1
    j = 0
    i= i + 1

n_img_rgb_corr = matplotlib.colors.hsv_to_rgb(n_img_hsv_shadow2)  # Umwandlung HSV in RGB
real_img_corr = numpytoimage(n_img_rgb_corr)
real_img_corr = PIL.ImageOps.invert(real_img_corr)

real_img_corr.show()

############ Speicherung des finalen Kartenausschnitts ############
###################################################################
path = 'C:\Users\student\PyCharm\Masterarbeit\output'
file = '\CORR' + ' (' + time.strftime("%d-%m-%y") + ')'
ending = '.png'
complete_path = path + file + ending

print(complete_path)
real_img_corr.save(complete_path)


# Ilmenau_streets = matplotlib.image.imread("C:\Users\student\PyCharm\Masterarbeit\output\CORR (09-11-17).png")  # Einlesen der Bilder
#
# #Ilmenau_streets = PIL.ImageOps.invert(Ilmenau_streets)
# n_img_noa = imagetonumpy(Ilmenau_streets)
# n_img_hsv = matplotlib.colors.rgb_to_hsv(n_img_noa)           # Umwandlung RGB in HSV
#
# #n_img_hsv = n_img_hsv_shadow2
#
# testarray_V_values = [0.5, 0.55]#, 0.6, 0.65, 0.7] #[0.7, 0.75, 0.8, 0.85, 0.9]    # 3. Kanal (HSV-Farbraum)
# testarray_S_values = 0.1                                        # 2. Kanal (HSV-Farbraum)
# #print(len(testarray_V_values))
#
# i = 0
# j = 0
# k = 0
# while k < len(testarray_V_values):
#     while i < n_img_hsv.shape[0]:         # Manipulation der HSV-Werte
#
#         while j < n_img_hsv.shape[1]:
#             if n_img_hsv[i, j, 2] < testarray_V_values[k]:
#                 n_img_hsv[i, j, :] = 0
#             else:
#                 n_img_hsv[i, j, 2] = n_img_hsv[i, j, 2]
#
#             # if n_img_hsv[i, j, 1] > testarray_S_values:
#             #     n_img_hsv[i, j, :] = 0
#             # else:
#             #     n_img_hsv[i, j, 1] = n_img_hsv[i, j, 1]
#
#             j = j + 1
#         j = 0
#         i= i + 1
#
#     """
#     Rücktransformation der numpy-Arrays in RGB-Bilder
#     """
#     n_img_rgb = matplotlib.colors.hsv_to_rgb(n_img_hsv)  # Umwandlung HSV in RGB
#
#     #n_img_rgb = n_img_rgb * 255
#     # print(n_img_rgb[:,:,:])
#
#
#     real_img_hsv = Image.fromarray(n_img_rgb.astype(np.uint8))  # TEST: Rückumwandlung von Array in Bild
#     # real_img_hsv.show()
#     real_img_hsv_inv = PIL.ImageOps.invert(real_img_hsv)    # Erzeugen des Positiv-Bildes
#     real_img_hsv_inv.show()
#
#     ###########
#
#     path = 'C:\Users\student\PyCharm\Masterarbeit\output'
#     file = '\V=' + str(testarray_V_values[k]) + ', ' + 'S=' + str(testarray_S_values) + ' (' + time.strftime("%d-%m-%y") + ')'
#     ending = '.png'
#     complete_path = path + file + ending
#
#     # path = 'C:\Users\student\PyCharm\Masterarbeit\output'
#     # file = '\V=' + str(testarray_V_values[k]) + ', ' + ' (' + time.strftime("%d-%m-%y") + ')'
#     # complete_path = path + file + ending
#
#     print(complete_path)
#     real_img_hsv_inv.save(complete_path)
#
#     ###########
#
#     i = 0
#     j = 0
#     k = k + 1





