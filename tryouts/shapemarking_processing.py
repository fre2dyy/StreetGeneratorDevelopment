import cv2
import numpy as np
from numpy import genfromtxt
import math
import csv


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


#===========================================================================================
# READ .CSVs
#===========================================================================================

datapath_csvs = "C:\Users\student\PyCharm\Masterarbeit\output\Test\Templates\orig\csvs";
ending_csvs = ".csv"

# READ ARROWS
arrows = []

# arrow_s
name_template = "\Arrow_s_orig"
my_data = genfromtxt(datapath_csvs + name_template + ending_csvs, delimiter=',')

arrow_s = np.reshape(my_data, (len(my_data)/2, 1, 2))
arrows.append(arrow_s)
print len(my_data)/2
#-------------------------------------------------------------------------------------------
# arrow_l
name_template = "\Arrow_l_orig"
my_data = genfromtxt(datapath_csvs + name_template + ending_csvs, delimiter=',')


arrow_l = np.reshape(my_data, (len(my_data)/2, 1, 2))
arrows.append(arrow_l)
print len(my_data)/2
#-------------------------------------------------------------------------------------------
# arrow_r
name_template = "\Arrow_r_orig"
my_data = genfromtxt(datapath_csvs + name_template + ending_csvs, delimiter=',')


arrow_r = np.reshape(my_data, (len(my_data)/2, 1, 2))
arrows.append(arrow_r)
print len(my_data)/2
#-------------------------------------------------------------------------------------------
# arrow_sl
name_template = "\Arrow_sl_orig"
my_data = genfromtxt(datapath_csvs + name_template + ending_csvs, delimiter=',')


arrow_sl = np.reshape(my_data, (len(my_data)/2, 1, 2))
arrows.append(arrow_sl)
print len(my_data)/2
#-------------------------------------------------------------------------------------------
# arrow_sr
name_template = "\Arrow_sr_orig"
my_data = genfromtxt(datapath_csvs + name_template + ending_csvs, delimiter=',')


arrow_sr = np.reshape(my_data, (len(my_data)/2, 1, 2))
arrows.append(arrow_sr)
print len(my_data)/2
#-------------------------------------------------------------------------------------------
# arrow_lr
name_template = "\Arrow_lr_orig"
my_data = genfromtxt(datapath_csvs + name_template + ending_csvs, delimiter=',')


arrow_lr = np.reshape(my_data, (len(my_data)/2, 1, 2))
arrows.append(arrow_lr)
print len(my_data)/2
#-------------------------------------------------------------------------------------------



#===================
# READ DATA
#===================

# datapath_a = "C:\Users\student\PyCharm\Masterarbeit\output\Test\Templates\orig\pictures";
#
datapath_picture = "C:\Users\student\PyCharm\Masterarbeit\output\Test\Templates";
#
# name_template = "\Arrow_r_orig"
# ending_picture = ".png"
#
# # first template
# a1 = cv2.imread(datapath_a + name_template + ending_picture)
#
# #a1 = cv2.imread(datapath_picture+"\TEST (09-10-17)_template_big_mirror_rot.png");



# second template
b = cv2.imread(datapath_picture+"\TEST (09-10-17)_template3.png");    # to be tested shape

#orig = [a1, a2, a3, a4, a5, a6]     # original shapes
#orig_array = np.asarray(orig)    # original shapes as array
#print(orig_array)

# DATA RESOLUTION
#height_a, width_a = orig_array.shape[:2]
height_b, width_b = b.shape[:2]
# print "============="
# #print(height_a, width_a)
# print height_b, width_b
# print "============="
# #print(np.shape(orig_array[0]))
# print np.shape(a1)
# print "============="

# threshold picture
imgray_b = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
ret_b,thresh_b = cv2.threshold(imgray_b,127,255,0)

# FIND CONTOURS
_, cb, _ = cv2.findContours(thresh_b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# ANGLE CALCULATION
rows_b,cols_b = b.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cb[0], cv2.DIST_L2,0,0.01,0.01)
lefty_b = int((-x*vy/vx) + y)
righty_b = int(((cols_b-x)*vy/vx)+y)
cv2.line(b,(cols_b-1,righty_b),(0,lefty_b),(0,255,0),2)

A_b = (0, 1)
B_b = (cols_b-1, righty_b-lefty_b)

print "Winkel Template 2: ", 180-angle_between(A_b, B_b)

# DRAW CONTOURS
# cv2.drawContours(b, cb, -1, (0,255,0), 1)
# cv2.imshow('2', b)



# ===============================================================================
# LOOP to campare with "arrows"

i = 0

while i < len(arrows):
    # ============================================================
    # CALCULATE HAUSDORFF AND SHAPECONTEXT DISTANCE
    # ============================================================
    hd = cv2.createHausdorffDistanceExtractor()
    sd = cv2.createShapeContextDistanceExtractor()

    d1 = hd.computeDistance(arrows[i], cb[0])
    d2 = sd.computeDistance(arrows[i], cb[0])

    # ============================================================
    # PRINT HAUSDORFF AND SHAPECONTEXT DISTANCE
    # ============================================================
    print "\n------------------------------------ \nHAUSDORFF AND SHAPECONTEXT DISTANCE (ARROW: ", i+1, "\n------------------------------------"
    print "hausdorff:    ", d1, "\nshapecontext: ", d2

    i = i + 1




# imgray_a = cv2.cvtColor(arrows[0],cv2.COLOR_BGR2GRAY)
# ret_a,thresh_a = cv2.threshold(imgray_a,127,255,0)
#
#
# #===================
# # FIND CONTOURS
# #===================
# _, ca, _ = cv2.findContours(thresh_a, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


#===================
# ANGLE CALCULATION (clockwise, startpoint = (0 / 1)
#===================

# first template
# rows_a,cols_a = arrows[0].shape[:2]
# [vx,vy,x,y] = cv2.fitLine(ca[0], cv2.DIST_L2,0,0.01,0.01)
# lefty_a = int((-x*vy/vx) + y)
# righty_a = int(((cols_a-x)*vy/vx)+y)
# cv2.line(a1,(cols_a-1,righty_a),(0,lefty_a),(0,255,0),2)
#
# A_a = (0, 1)
# B_a = (cols_a-1, righty_a-lefty_a)
#
#
# print "Winkel Template 1: ", 180-angle_between(A_a, B_a)

# print cols_a, righty_a, "0", lefty_a

# #===================
# # .PNG in .CSV
# #===================
#
# # print "nshape: ", np.shape(ca[0])#, np.shape(cb[0])
# # print(ca)
# ca = np.asarray(ca)
# # print(ca[0])
# # print(np.shape(ca[0]))
#
# shape_ca = np.shape(ca[0])
# # print shape_ca[0]
#
# datapath_csvs = "C:\Users\student\PyCharm\Masterarbeit\output\Test\Templates\orig\csvs";
# ending_csvs = ".csv"
#
# ca.tofile(datapath_csvs + name_template + ending_csvs, sep=',', format='%10.0f')


#===================
# DRAW CONTOURS
#===================
# cv2.drawContours(a1, ca, -1, (0,255,0), 1)
# cv2.imshow('1', a1)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



