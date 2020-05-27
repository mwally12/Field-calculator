#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:44:13 2020

@author: mbejarano
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import ticker
from numpy import ma
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import pandas as pd


def B(x, y, xm, ym, Mx, My):      
    """
    Function to calculate the external field due to magnetic dipole moments
    See link for formula: https://en.wikipedia.org/wiki/Dipole
    Parameters:
    x - x coord of inquired point
    y - y coord of inquired point
    xm and ym - coordinates of magnetic cell
    Mx and My - magnetization components of the magnetic cells
    """        
    mu_0 = 4*np.pi*10**-7                                   #Units: H/m
    fac = mu_0/(4*np.pi)                                    #Multiplicative factor of formula
    Bx=0
    By=0
    for row in range(len(Mx)):                              #2 for loops for iterating over magnet grid
        for col in range(len(Mx[row])):
            r_x=x-xm[row][col]                              #calculate displacement components bet. eval points and magn cell
            r_y=y-ym[row][col]                              #r_x holds x components, r_y y components
            magn=np.hypot(r_x,r_y)                          #grid with the magnitude of the r vectors
            r=np.array([r_x/magn,r_y/magn])                 #unitary r matrix
            Bx += fac * (3* (Mx[row][col]*r[0]+My[row][col]*r[1])*r[0]-Mx[row][col])/(magn**3)
            By += fac * (3* (Mx[row][col]*r[0]+My[row][col]*r[1])*r[1]-My[row][col])/(magn**3)

    return Bx,By


# =============================================================================
# Extracting magnetization vector and field values from simulation from csv
# =============================================================================

magnetization = pd.read_csv('/Users/mbejarano/Documents/Work/Simulations/text_square.out/relax.csv',header=None).as_matrix()

new_d = np.reshape(magnetization,(3,10,10))
mx= new_d[0]
my= new_d[1]
mz= new_d[2]
mx_dim, my_dim = mx.shape

field = pd.read_csv('/Users/mbejarano/Documents/Work/Simulations/sim_field/B_demag_square.csv',header=None).as_matrix()
new_field = np.reshape(field,(3,100,100))
Bx_sim= new_field[0]
By_sim= new_field[1]
Bz_sim= new_field[2]

nx_sim,ny_sim = Bx_sim.shape
XMAX_sim, YMAX_sim= 50,50
x_sim = np.linspace(-XMAX_sim, XMAX_sim, nx_sim)        # 1D vector of x coordinates
y_sim = np.linspace(-YMAX_sim, YMAX_sim, ny_sim)        # 1D vector of y coordinates


# =============================================================================
# Creation of evaluation space
# Note: have to be careful of not putting an eval point inside the magnet, or else the formula will result in a very
# big value (measuring field inside the magnet!). So choose amount of points and max coordinates carefully.
# =============================================================================

unit_size=10**-6                             #unit: um
nx, ny,nz = 100, 100, 1                         #grid size x vs y
XMAX, YMAX, ZMAX = 50*unit_size, 50*unit_size, 1*unit_size   #max values for grid  
x = np.linspace(-XMAX, XMAX, nx)        # 1D vector of x coordinates
y = np.linspace(-YMAX, YMAX, ny)        # 1D vector of y coordinates
z = np.linspace(-ZMAX, ZMAX, nz)        # 1D vector of y coordinates
X, Y = np.meshgrid(x, y)                # X grid with x coordinates; Y grid with y coordinates
Y2, Z2, X2 = np.meshgrid(y,z,x)                # X grid with x coordinates; Y grid with y coordinates


x2 = np.linspace(-50, 50, nx)        # 1D vector of x coordinates
y2 = np.linspace(-50, 50, ny)        # 1D vector of y coordinates

# =============================================================================
# Creation of magnet
# =============================================================================

length_mag=10*unit_size                              #Dimensions for a magnet
width_mag=10*unit_size
height_mag=50*(10**-9)
Vol=length_mag*width_mag*height_mag                 #Volume of the magnet
xm,ym,zm=mx_dim,my_dim,1                                           #grid size for magnet
Vol_unit=Vol/(xm*ym)
Vol_unit2=Vol/(xm*ym*zm)
XM_Max, YM_Max, ZM_Max=length_mag/2, width_mag/2, height_mag/2
m_x_coord = np.linspace(-XM_Max, XM_Max, xm)        # 1D vector of x coordinates
m_y_coord = np.linspace(-YM_Max, YM_Max, ym)        # 1D vector of y coordinates
m_z_coord = np.linspace(-ZM_Max, ZM_Max, zm)        # 1D vector of y coordinates
XM, YM = np.meshgrid(m_x_coord, m_y_coord)          # X/Y grid with magnet cell coordinates
YM2, ZM2, XM2 = np.meshgrid(m_y_coord,m_z_coord, m_x_coord)          # X/Y grid with magnet cell coordinates
Ms = 810000                                         #Magnetization saturation
#mx=np.ones((xm,ym))                                 #unitary magnetization matrix of mx components
#my=np.zeros((xm,ym))                                #unitary magnetization matrix of my components
Mx=Ms*Vol_unit*mx                                   #Magnetization X component
My=Ms*Vol_unit*my                                   #Magnetization Y component




# =============================================================================
# Calling function for calculating the field at ALL grid points
# =============================================================================

Bx,By = B(X,Y,XM,YM,Mx,My)


# =============================================================================
# Extracting the field at one particular point
# =============================================================================

Bx_ext=0.037
print("The Bx component at that point is {:e}".format(Bx[61][62]))
print("The By component at that point is {:e}".format(By[61][62]))

print("The Bx component from the simulation at that point is {:e}".format(Bx_sim[61][62]))
print("The By component from the simulation at that point is {:e}".format(By_sim[61][62]))


# =============================================================================
# Plotting settings
# Note:Many things are commented out as I was experimenting with the type of display
# =============================================================================

fig = plt.figure(figsize=(11, 7))
gs = gridspec.GridSpec(nrows=2, ncols=3)
cmap = plt.cm.RdBu_r

# =============================================================================
# Contour plot of Bx
# =============================================================================
ax0 = fig.add_subplot(gs[0, 0])
norm=colors.SymLogNorm(linthresh=0.000001,linscale=1, vmin=-0.1, vmax=0.1)
#cs1 = ax0.contourf(x, y, Bx, norm=colors.SymLogNorm(linthresh=0.1,linscale=1, vmin=Bx.min(), vmax=Bx.max()),cmap='RdBu')
#cs1 = ax0.contourf(x, y, Bx,  norm=MidpointNormalize(midpoint=0.,vmin=Bx.min(), vmax=Bx.max()),cmap='RdBu_r')
cs1 = ax0.contourf(x2, y2, Bx,locator=ticker.SymmetricalLogLocator(base=10, linthresh=0.000001),norm=norm,cmap='RdBu_r')
#cs1 = ax0.contourf(x, y, Bx,norm=colors.SymLogNorm(linthresh=0.1,linscale=0.1,vmin=Bx.min(), vmax=Bx.max()),cmap='RdBu_r')
ax0.set_title("Bx")
#ax0.set_xlabel("x ({})".format(units[unit_size]))
#ax0.set_ylabel("y ({})".format(units[unit_size]))
ax0.set_xlabel(r'x ($\mu$m)')
ax0.set_ylabel(r'y ($\mu$m)')
ax0.set_xlim(-50, 50)
ax0.set_ylim(-50, 50)
#ax0.ticklabel_format(axis='both',style='sci',scilimits=(0,0),useMathText=True)
cbar1=fig.colorbar(cs1,ax=ax0,extend='both')
cbar1.set_label('Log(Bx)',rotation=270)
ax0.add_patch(Rectangle((-5,-5),10,10, color='k', zorder=100))
ax0.add_patch(Circle((12.6,11),radius=1, color='g', zorder=100))

# =============================================================================
# Contour plot of By
# =============================================================================

ax1 = fig.add_subplot(gs[0, 1])
norm2=colors.SymLogNorm(linthresh=0.000001,linscale=1, vmin=-0.1, vmax=0.1)
#cs2 = ax1.contourf(x, y, By, locator=ticker.LogLocator(),cmap='RdBu_r')
cs2 = ax1.contourf(x2, y2, By, locator=ticker.SymmetricalLogLocator(base=10, linthresh=0.000001), norm=norm2,cmap='RdBu_r')
cbar2=fig.colorbar(cs2, ax=ax1)
cbar2.set_label('Log(By)',rotation=270)
ax1.set_title("By")
#ax1.set_xlabel("x ({})".format(units[unit_size]))
#ax1.set_ylabel("y ({})".format(units[unit_size]))
ax1.set_xlabel(r'x ($\mu$m)')
ax1.set_ylabel(r'y ($\mu$m)')
ax1.set_xlim(-50, 50)
ax1.set_ylim(-50, 50)
ax1.add_patch(Rectangle((-5,-5),10,10, color='k', zorder=100))
ax1.add_patch(Circle((12.6,11),radius=1, color='g', zorder=100))

# =============================================================================
# Vector field plot
# =============================================================================
# Plot the streamlines with an appropriate colormap and arrow style
color = 2* np.log(np.hypot(Bx, By))
ax2 = fig.add_subplot(gs[0,2])
strm=ax2.streamplot(x2, y2, Bx, By, color=color, linewidth=1, cmap=plt.cm.inferno,density=0.7, arrowstyle='->', arrowsize=1.5)
cbar3=fig.colorbar(strm.lines,ax=ax2,extend='both')
cbar3.set_label(r'Log($\vert$B$\vert$)')
ax2.set_title("Vector field")
ax2.set_xlabel(r'x ($\mu$m)')
ax2.set_ylabel(r'y ($\mu$m)')
ax2.set_xlim(-50, 50)
ax2.set_ylim(-50, 50)
ax2.set_aspect('equal')
ax2.add_patch(Rectangle((-5,-5),10,10, color='k', zorder=100))
ax2.add_patch(Circle((12.6,11),radius=1, color='g', zorder=100))

# =============================================================================
# Contour plot of Bx simulation
# =============================================================================
ax3 = fig.add_subplot(gs[1, 0])
norm3=colors.SymLogNorm(linthresh=0.000001,linscale=1, vmin=Bx_sim.min(), vmax=Bx_sim.max())
#cs1 = ax0.contourf(x, y, Bx, norm=colors.SymLogNorm(linthresh=0.1,linscale=1, vmin=Bx.min(), vmax=Bx.max()),cmap='RdBu')
#cs1 = ax0.contourf(x, y, Bx,  norm=MidpointNormalize(midpoint=0.,vmin=Bx.min(), vmax=Bx.max()),cmap='RdBu_r')
cs3 = ax3.contourf(x_sim, y_sim, Bx_sim,locator=ticker.SymmetricalLogLocator(base=10, linthresh=0.000001),norm=norm3,cmap='RdBu_r')
#cs1 = ax0.contourf(x, y, Bx,norm=colors.SymLogNorm(linthresh=0.1,linscale=0.1,vmin=Bx.min(), vmax=Bx.max()),cmap='RdBu_r')
ax3.set_title("Bx")
#ax0.set_xlabel("x ({})".format(units[unit_size]))
#ax0.set_ylabel("y ({})".format(units[unit_size]))
ax3.set_xlabel(r'x ($\mu$m)')
ax3.set_ylabel(r'y ($\mu$m)')
ax3.set_xlim(-XMAX_sim, XMAX_sim)
ax3.set_ylim(-YMAX_sim, YMAX_sim)
#ax0.ticklabel_format(axis='both',style='sci',scilimits=(0,0),useMathText=True)
cbar4=fig.colorbar(cs3,ax=ax3)
cbar4.set_label('Log(Bx)',rotation=270)
ax3.add_patch(Rectangle((-5,-5),10,10, color='k', zorder=100))
ax3.add_patch(Circle((12.6,11),radius=1, color='g', zorder=100))

# =============================================================================
# Contour plot of By
# =============================================================================

ax4 = fig.add_subplot(gs[1, 1])
norm4=colors.SymLogNorm(linthresh=0.000001,linscale=1, vmin=By_sim.min(), vmax=By_sim.max())
cs4 = ax4.contourf(x_sim, y_sim, By_sim, locator=ticker.SymmetricalLogLocator(base=10, linthresh=0.000001), norm=norm4,cmap='RdBu_r')
cbar5=fig.colorbar(cs4, ax=ax4)
cbar5.set_label('Log(By)',rotation=270)
ax4.set_title("By")
ax4.set_xlabel(r'x ($\mu$m)')
ax4.set_ylabel(r'y ($\mu$m)')
ax4.set_xlim(-XMAX_sim, XMAX_sim)
ax4.set_ylim(-YMAX_sim, YMAX_sim)
ax4.add_patch(Rectangle((-5,-5),10,10, color='k', zorder=100))
ax4.add_patch(Circle((12.6,11),radius=1, color='g', zorder=100))

ax5 = fig.add_subplot(gs[1, 2])
color2 = 2* np.log(np.hypot(Bx_sim, By_sim))
strm=ax5.streamplot(x_sim, y_sim, Bx_sim, By_sim, color=color2, linewidth=1, cmap=plt.cm.inferno,density=0.7, arrowstyle='->', arrowsize=1.5)
cbar6=fig.colorbar(strm.lines,ax=ax5,extend='both')
cbar6.set_label(r'Log($\vert$B$\vert$)')
ax5.set_title("Vector field")
ax5.set_xlabel(r'x ($\mu$m)')
ax5.set_ylabel(r'y ($\mu$m)')
ax5.set_xlim(-XMAX_sim, XMAX_sim)
ax5.set_ylim(-YMAX_sim, YMAX_sim)
ax5.set_aspect('equal')
ax5.add_patch(Rectangle((-5,-5),10,10, color='k', zorder=100))
ax5.add_patch(Circle((12.6,11),radius=1, color='g', zorder=100))

fig.tight_layout()



fig2 = plt.figure(figsize=(8, 3.5))
gs2 = gridspec.GridSpec(nrows=1, ncols=2)


Bx_diff=np.absolute(Bx-Bx_sim)
By_diff=np.absolute(By-By_sim)

ax6 = fig2.add_subplot(gs2[0, 0])
norm5=colors.SymLogNorm(linthresh=0.000001,linscale=1, vmin=0, vmax=0.1)
cs5 = ax6.contourf(x_sim, y_sim, Bx_diff, locator=ticker.SymmetricalLogLocator(base=10, linthresh=0.000001), norm=norm5,cmap='RdBu_r')
cbar7=fig2.colorbar(cs5, ax=ax6)
cbar7.set_label('Log(Bx)',rotation=270)
ax6.set_title("$\Vert \Delta Bx \Vert$")
ax6.set_xlabel(r'x ($\mu$m)')
ax6.set_ylabel(r'y ($\mu$m)')
ax6.set_xlim(-XMAX_sim, XMAX_sim)
ax6.set_ylim(-YMAX_sim, YMAX_sim)
ax6.add_patch(Rectangle((-5,-5),10,10, color='k', zorder=100))
ax6.add_patch(Circle((12.6,11),radius=1, color='g', zorder=100))

ax7 = fig2.add_subplot(gs2[0, 1])
norm6=colors.SymLogNorm(linthresh=0.000001,linscale=1, vmin=0, vmax=0.1)
cs6 = ax7.contourf(x_sim, y_sim, By_diff, locator=ticker.SymmetricalLogLocator(base=10, linthresh=0.000001), norm=norm6,cmap='RdBu_r')
cbar8=fig2.colorbar(cs6, ax=ax7)
cbar8.set_label('Log(By)',rotation=270)
ax7.set_title("$\Vert \Delta By \Vert$")
ax7.set_xlabel(r'x ($\mu$m)')
ax7.set_ylabel(r'y ($\mu$m)')
ax7.set_xlim(-XMAX_sim, XMAX_sim)
ax7.set_ylim(-YMAX_sim, YMAX_sim)
ax7.add_patch(Rectangle((-5,-5),10,10, color='k', zorder=100))
ax7.add_patch(Circle((12.6,11),radius=1, color='g', zorder=100))

fig2.tight_layout()
plt.show()
