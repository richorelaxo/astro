# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:45:16 2023
@author: Richard White

2-body Celestial Mechanics
pygame

F = G mM/r^2
"""

# libraries
import numpy as np
import matplotlib.pyplot as plt

#%% static init
# time
dt = 0.1
G = 6.67430e-11


#%% Functions

# Radius/agnle between points
def radiDeg(x, y):
    rx = x[1] - x[0]
    ry = y[1] - y[0]
    r = np.sqrt(rx**2 + ry**2)
    rang = np.arccos(np.dot([1,0], [rx,ry]/r))
    return r, rang

# Gravitational force - TBC
def forceXY(x, y, m, M):
    G = 6.67430e-11
    rs = radiDeg(x, y)
    r = rs[0]
    ang = rs[1]
    F = G*m*M/r**2
    Fx = np.abs(F * np.cos(ang))
    Fy = np.abs(F * np.sin(ang))
    return [Fx, Fy]

# Orientation
def orient(vec):
    ort = np.array([0,0])
    
    # orientation
    if vec[1] > vec[0]:
        ort = np.array([1,-1])
    else:
        ort = np.array([-1,1])
    return ort

#%% Sim 

# sim init
itrs = 1000

# masses
m = np.array([1e15, 1e15])

# positions 
px = np.array([100, 200])
py = px

# velocities
dx = np.array([5, -4])
dy = np.array([-5, 5])

# recorders
x_pos = np.zeros([itrs,2])
y_pos = np.zeros([itrs,2])
x_vel = np.zeros([itrs,2])
y_vel = np.zeros([itrs,2])
x_acc = np.zeros([itrs,2])
y_acc = np.zeros([itrs,2])
f_rec = np.zeros([itrs,2])
x_pos[0] = px
y_pos[0] = py

# Space loop
for i in range(1,itrs):
    # force
    F = forceXY(px, py, m[0], m[1])
    f_rec[i] = F
    
    # acceleration
    ax = F[0]*orient(px)/m
    ay = F[1]*orient(py)/m
    
    # velocity
    dx = dx + ax*dt
    dy = dy + ay*dt
    
    # position
    px = px + dx*dt
    py = py + dy*dt
    
    # recorders
    x_pos[i,:] = px
    y_pos[i,:] = py
    x_vel[i,:] = dx
    y_vel[i,:] = dy
    x_acc[i,:] = ax
    y_acc[i,:] = ay


# Plot
fig, ax0 = plt.subplots(1,1, figsize=(10,10))

ax0.grid()
# M
ax0.plot(x_pos[:,1], y_pos[:,1], color='r')
ax0.quiver(x_pos[:,1], y_pos[:,1], x_acc[:,1], y_acc[:,1], color='m',
          width=0.002);
# m
ax0.plot(x_pos[:,0], y_pos[:,0], color='b')
ax0.quiver(x_pos[:,0], y_pos[:,0], x_acc[:,0], y_acc[:,0], color='c',
          width=0.002);

ax0.scatter(x_pos[:,0] + (np.diff(x_pos)/2).T, y_pos[:,0] + (np.diff(y_pos)/2).T,
           color='k', s=5)
if 0:
    cnt = [200,200]
    scale = 200
    ax0.set_xlim([cnt[0]-scale,cnt[0]+scale])
    ax0.set_ylim([cnt[1]-scale,cnt[1]+scale])

#%% Recorders
fig, ax = plt.subplots(1,1, figsize=(7,7))

ax.grid()
ax.plot(x_acc[:,1])
ax.plot(y_acc[:,1])





















    