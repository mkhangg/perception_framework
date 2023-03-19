import numpy as np
import math


#print(np.load('cam_matrix/mtx1.npy'))
m1 = np.load('cam_matrix/mtx1.npy')
m2 = np.load('cam_matrix/mtx2.npy')
m3 = np.load('cam_matrix/mtx3.npy')
m4 = np.load('cam_matrix/mtx4.npy')

print(m1)
print(m2)
print(m3)
print(m4)

m1, m2, m3, m4 = m1.flatten(), m2.flatten(), m3.flatten(), m4.flatten(),

print(math.dist(m1, m2))
print(math.dist(m1, m3))
print(math.dist(m1, m4))
