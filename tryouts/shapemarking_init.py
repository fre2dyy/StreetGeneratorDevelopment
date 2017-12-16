import cv2
import numpy as np
from numpy import genfromtxt
import math
import csv

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

#===================
# READ DATA
#===================
datapath_a = "C:\Users\student\PyCharm\Masterarbeit\output\Test\Templates\orig\pictures";

datapath_picture = "C:\Users\student\PyCharm\Masterarbeit\output\Test\Templates";

name_template = "\Arrow_s_orig"
ending_picture = ".png"

# first template
a1 = cv2.imread(datapath_a + name_template + ending_picture)

#a1 = cv2.imread(datapath_picture+"\TEST (09-10-17)_template_big_mirror_rot.png");



# second template
b = cv2.imread(datapath_picture+"\TEST (09-10-17)_template6.png");    # to be tested shape

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



imgray_a = cv2.cvtColor(a1,cv2.COLOR_BGR2GRAY)
ret_a,thresh_a = cv2.threshold(imgray_a,127,255,0)

imgray_b = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
ret_b,thresh_b = cv2.threshold(imgray_b,127,255,0)


#===================
# FIND CONTOURS
#===================
_, ca, _ = cv2.findContours(thresh_a, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
_, cb, _ = cv2.findContours(thresh_b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


#===================
# ANGLE CALCULATION (clockwise, startpoint = (0 / 1)
#===================

# first template
rows_a,cols_a = a1.shape[:2]
[vx,vy,x,y] = cv2.fitLine(ca[0], cv2.DIST_L2,0,0.01,0.01)
lefty_a = int((-x*vy/vx) + y)
righty_a = int(((cols_a-x)*vy/vx)+y)
cv2.line(a1,(cols_a-1,righty_a),(0,lefty_a),(0,255,0),2)


A_a = (0, 1)
B_a = (cols_a-1, righty_a-lefty_a)

rows_b,cols_b = b.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cb[0], cv2.DIST_L2,0,0.01,0.01)
lefty_b = int((-x*vy/vx) + y)
righty_b = int(((cols_b-x)*vy/vx)+y)
cv2.line(b,(cols_b-1,righty_b),(0,lefty_b),(0,255,0),2)


A_b = (0, 1)
B_b = (cols_b-1, righty_b-lefty_b)


print "Winkel Template 1: ", 180-angle_between(A_a, B_a)
print "Winkel Template 2: ", 180-angle_between(A_b, B_b)

# print cols_a, righty_a, "0", lefty_a

#===================
# .PNG in .CSV
#===================

# print "nshape: ", np.shape(ca[0])#, np.shape(cb[0])
# print(ca)
ca = np.asarray(ca)
# print(ca[0])
# print(np.shape(ca[0]))

shape_ca = np.shape(ca[0])
# print shape_ca[0]

datapath_csvs = "C:\Users\student\PyCharm\Masterarbeit\output\Test\Templates\orig\csvs";
ending_csvs = ".csv"

#ca.tofile(datapath_csvs + name_template + ending_csvs, sep=',', format='%10.0f')


#===================
# DRAW CONTOURS
#===================
cv2.drawContours(a1, ca, -1, (0,255,0), 1)
# cv2.imshow('1', a1)

cv2.drawContours(b, cb, -1, (0,255,0), 1)
# cv2.imshow('2', b)

cv2.waitKey(0)
cv2.destroyAllWindows()




# cont = cv2.drawContours(img, contours, -1, (0,255,0), 3)

#[float(i) for i in ca]

# np_ca = np.asarray(ca)#, np.int)
# np_cb = np.asarray(cb)#, np.int)

# print("----------------------------------")
# print("ca: ", np.shape(ca))
# print("cb: ", np.shape(cb))
# print("----------------------------------")
# print("np_ca: ", np.shape(np_ca))
# print("np_cb: ", np.shape(np_cb))
# print("----------------------------------")
# print("dtype(np_ca): ", np_ca.dtype)
# print("dtype(np_cb): ", np_cb.dtype)
# print("----------------------------------")


#print(ca)
#print(np_ca)
#print(type(np_ca))
#print(thresh_a)
#print(np_ca)
#print(type(np_ca))

#===================
# READ .CSV
#===================
my_data = genfromtxt(datapath_csvs + name_template + ending_csvs, delimiter=',')
#print(np.shape(my_data))
# print(my_data)

my_data_reshape = np.reshape(my_data, (shape_ca[0], 1, 2))
# print my_data_reshape
print shape_ca[0]
shape_cb = np.shape(cb[0])
print shape_cb[0]

#============================================================
# CALCULATE HAUSDORFF AND SHAPECONTEXT DISTANCE
#============================================================
hd = cv2.createHausdorffDistanceExtractor()
sd = cv2.createShapeContextDistanceExtractor()

d1 = hd.computeDistance(my_data_reshape,cb[0])
d2 = sd.computeDistance(my_data_reshape,cb[0])


#============================================================
# PRINT HAUSDORFF AND SHAPECONTEXT DISTANCE
#============================================================
print "\n------------------------------------ \nHAUSDORFF AND SHAPECONTEXT DISTANCE \n------------------------------------"
print "hausdorff:    ", d1, "\nshapecontext: ", d2