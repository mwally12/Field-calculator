#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:09:57 2020

@author: mbejarano
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import ticker
from numpy import ma
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d


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



def B2(x, y, z, xm, ym, zm, Mx, My, Mz):      
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
    Bz=0
    for depth in range(len(Mx)):                              #iterating over depth
        for row in range(len(Mx[depth])):
            for col in range(len(Mx[depth][row])):
                r_x=x-xm[depth][row][col]                              #calculate displacement components bet. eval points and magn cell
                r_y=y-ym[depth][row][col]                              #r_x holds x components, r_y y components
                r_z=z-zm[depth][row][col]                              #r_x holds x components, r_y y components
                magn=np.sqrt(r_x**2+r_y**2+r_z**2)
                #r=np.array([r_z/magn, r_x/magn,r_y/magn])                 #unitary r matrix
                rx=r_x/magn
                ry=r_y/magn
                rz=r_z/magn
                Bx += fac * (3* (Mx[depth][row][col]*rx+My[depth][row][col]*ry+Mz[depth][row][col]*rz)*rx-Mx[depth][row][col])/(magn**3)
                By += fac * (3* (Mx[depth][row][col]*rx+My[depth][row][col]*ry+Mz[depth][row][col]*rz)*ry-My[depth][row][col])/(magn**3)
                Bz += fac * (3* (Mx[depth][row][col]*rx+My[depth][row][col]*ry+Mz[depth][row][col]*rz)*rz-Mz[depth][row][col])/(magn**3)
#                Bx += fac * (3* (Mx[depth][row][col]*r[1]+My[depth][row][col]*r[2]+Mz[depth][row][col]*r[0])*r[1]-Mx[depth][row][col])/(magn**3)
#                By += fac * (3* (Mx[depth][row][col]*r[1]+My[depth][row][col]*r[2]+Mz[depth][row][col]*r[0])*r[2]-My[depth][row][col])/(magn**3)
#                Bz += fac * (3* (Mx[depth][row][col]*r[1]+My[depth][row][col]*r[2]+Mz[depth][row][col]*r[0])*r[0]-Mz[depth][row][col])/(magn**3)

    return Bx,By,Bz

# =============================================================================
# Creation of evaluation space
# Note: have to be careful of not putting an eval point inside the magnet, or else the formula will result in a very
# big value (measuring field inside the magnet!). So choose amount of points and max coordinates carefully.
# =============================================================================

# Create dictionary of units and value
units = {
        1:"m",
        10**-1:"dm",
        10**-2:"cm",
        10**-3: "mm",
        10**-6: '\u03BCm'
        }


unit_size=10**-6                        #unit: um
nx, ny,nz = 50, 51, 11                         #grid size x vs y
XMAX, YMAX, ZMAX = 225*unit_size, 225*unit_size, 10*unit_size   #max values for grid  
x = np.linspace(-XMAX, XMAX, nx)        # 1D vector of x coordinates
y = np.linspace(-YMAX, YMAX, ny)        # 1D vector of y coordinates
z = np.linspace(-ZMAX, ZMAX, nz)        # 1D vector of y coordinates
X, Y = np.meshgrid(x, y)                # X grid with x coordinates; Y grid with y coordinates
Y2, Z2, X2 = np.meshgrid(y,z,x)                # X grid with x coordinates; Y grid with y coordinates


# =============================================================================
# Creation of magnet
# =============================================================================

length_mag=8*unit_size                              #Dimensions for a cube-shaped magnet
width_mag=2*unit_size
height_mag=50*10**-9
Vol=length_mag*width_mag*height_mag                 #Volume of the magnet
xm,ym,zm=10,10,1                                           #grid size for magnet
Vol_unit=Vol/(xm*ym)
Vol_unit2=Vol/(xm*ym*zm)
XM_Max, YM_Max, ZM_Max=length_mag/2, width_mag/2, height_mag/2
m_x_coord = np.linspace(-XM_Max, XM_Max, xm)        # 1D vector of x coordinates
m_y_coord = np.linspace(-YM_Max, YM_Max, ym)        # 1D vector of y coordinates
m_z_coord = np.linspace(-ZM_Max, ZM_Max, zm)        # 1D vector of y coordinates
XM, YM = np.meshgrid(m_x_coord, m_y_coord)          # X/Y grid with magnet cell coordinates
YM2, ZM2, XM2 = np.meshgrid(m_y_coord,m_z_coord, m_x_coord)          # X/Y grid with magnet cell coordinates
Ms = 810000                                         #Magnetization saturation
mx=np.ones((xm,ym))                                 #unitary magnetization matrix of mx components
my=np.zeros((xm,ym))                                #unitary magnetization matrix of my components
Mx=Ms*Vol_unit*mx                                   #Magnetization X component
My=Ms*Vol_unit*my                                   #Magnetization Y component

mx2=np.ones((zm,xm,ym))                                 #unitary magnetization matrix of mx components
my2=np.zeros((zm,xm,ym))                                #unitary magnetization matrix of my components
mz2=np.zeros((zm,xm,ym))                                #unitary magnetization matrix of my components
Mx2=Ms*Vol_unit2*mx2                                   #Magnetization X component
My2=Ms*Vol_unit2*my2                                   #Magnetization Y component
Mz2=Ms*Vol_unit2*mz2                                   #Magnetization Y component


# =============================================================================
# Calling function for calculating the field at ALL grid points
# =============================================================================

Bx,By = B(X,Y,XM,YM,Mx,My)
Bx2,By2,Bz = B2(X2,Y2,Z2,XM2,YM2,ZM2,Mx2,My2,Mz2)

# =============================================================================
# Extracting the field at one particular point
# =============================================================================

#step_size_x=XMAX*2/(nx-1)
#
#x_inspect=step_size_x/2+2*step_size_x                               #Coordinates to inspect the field at this position
#y_inspect=4*unit_size                               #This point must be at the grid, otherwise there will be an error
#x_index=np.where(x==x_inspect)
#y_index=np.where(y==y_inspect)

#Bx_point=Bx[x_index[0][0]][y_index[0][0]]
#By_point=By[x_index[0][0]][y_index[0][0]]
Bx_ext=0.037
B_magn=np.sqrt(Bx2[5][25][25]**2+By2[5][25][25]**2+Bz[5][25][25]**2)
print("The Bx component at that point is {:e}".format(Bx2[5][25][25]))
print("The By component at that point is {:e}".format(By2[5][25][25]))
print("The Bz component at that point is {:e}".format(Bz[5][25][25]))
print("The B magnitude at that point is {:e}".format(B_magn))

# =============================================================================
# Plotting settings
# Note:Many things are commented out as I was experimenting with the type of display
# =============================================================================

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(nrows=2, ncols=2)
cmap = plt.cm.RdBu_r

# =============================================================================
# Contour plot of Bx
# =============================================================================
ax0 = fig.add_subplot(gs[0, 0])
norm=colors.SymLogNorm(linthresh=0.000001,linscale=1, vmin=Bx.min(), vmax=Bx.max())
#cs1 = ax0.contourf(x, y, Bx, norm=colors.SymLogNorm(linthresh=0.1,linscale=1, vmin=Bx.min(), vmax=Bx.max()),cmap='RdBu')
#cs1 = ax0.contourf(x, y, Bx,  norm=MidpointNormalize(midpoint=0.,vmin=Bx.min(), vmax=Bx.max()),cmap='RdBu_r')
cs1 = ax0.contourf(x, y, Bx,locator=ticker.SymmetricalLogLocator(base=10, linthresh=0.000001),norm=norm,cmap='RdBu_r')
#cs1 = ax0.contourf(x, y, Bx,norm=colors.SymLogNorm(linthresh=0.1,linscale=0.1,vmin=Bx.min(), vmax=Bx.max()),cmap='RdBu_r')
ax0.set_title("Bx")
#ax0.set_xlabel("x ({})".format(units[unit_size]))
#ax0.set_ylabel("y ({})".format(units[unit_size]))
ax0.set_xlabel(r'$x$ (m)')
ax0.set_ylabel(r'$y$ (m)')
ax0.set_xlim(-XMAX/2, XMAX/2)
ax0.set_ylim(-YMAX/2, YMAX/2)
#ax0.ticklabel_format(axis='both',style='sci',scilimits=(0,0),useMathText=True)
cbar1=fig.colorbar(cs1,ax=ax0)
cbar1.set_label('Log(Bx)',rotation=270)
ax0.add_patch(Rectangle((-XM_Max,-YM_Max),length_mag,width_mag, color='k', zorder=100))

# =============================================================================
# Contour plot of By
# =============================================================================

ax1 = fig.add_subplot(gs[0, 1])
norm2=colors.SymLogNorm(linthresh=0.000001,linscale=1, vmin=By.min(), vmax=By.max())
#cs2 = ax1.contourf(x, y, By, locator=ticker.LogLocator(),cmap='RdBu_r')
cs2 = ax1.contourf(x, y, By, locator=ticker.SymmetricalLogLocator(base=10, linthresh=0.000001), norm=norm2,cmap='RdBu_r')
cbar2=fig.colorbar(cs2, ax=ax1)
cbar2.set_label('Log(By)',rotation=270)
ax1.set_title("By")
#ax1.set_xlabel("x ({})".format(units[unit_size]))
#ax1.set_ylabel("y ({})".format(units[unit_size]))
ax1.set_xlabel(r'$x$ (m)')
ax1.set_ylabel(r'$y$ (m)')
ax1.set_xlim(-XMAX, XMAX)
ax1.set_ylim(-YMAX, YMAX)
ax1.add_patch(Rectangle((-XM_Max,-YM_Max),length_mag,width_mag, color='k', zorder=100))

# =============================================================================
# Vector field plot
# =============================================================================
# Plot the streamlines with an appropriate colormap and arrow style
color = 2* np.log(np.hypot(Bx, By))
ax2 = fig.add_subplot(gs[1,:])
strm=ax2.streamplot(x, y, Bx, By, color=color, linewidth=1, cmap=plt.cm.inferno,density=1, arrowstyle='->', arrowsize=1.5)
cbar3=fig.colorbar(strm.lines,ax=ax2,extend='both')
cbar3.set_label(r'Log($\vert$B$\vert$)')
ax2.set_title("Vector field")
ax2.set_xlabel(r'$x$ (m)')
ax2.set_ylabel(r'$y$ (m)')
ax2.set_xlim(-XMAX, XMAX)
ax2.set_ylim(-YMAX, YMAX)
ax2.set_aspect('equal')
ax2.add_patch(Rectangle((-XM_Max,-YM_Max),length_mag,width_mag, color='k', zorder=100))

fig.tight_layout()
plt.show()

#######
# Second figure

fig2 = plt.figure(figsize=(10, 10))

# =============================================================================
# Contour plot of Bx
# =============================================================================
ax3 = fig2.add_subplot(gs[0, 0])
norm3=colors.SymLogNorm(linthresh=0.000001,linscale=1, vmin=Bx2.min(), vmax=Bx2.max())
#cs1 = ax0.contourf(x, y, Bx, norm=colors.SymLogNorm(linthresh=0.1,linscale=1, vmin=Bx.min(), vmax=Bx.max()),cmap='RdBu')
#cs1 = ax0.contourf(x, y, Bx,  norm=MidpointNormalize(midpoint=0.,vmin=Bx.min(), vmax=Bx.max()),cmap='RdBu_r')
cs3 = ax3.contourf(x, y, Bx2[5],locator=ticker.SymmetricalLogLocator(base=10, linthresh=0.000001),norm=norm3,cmap='RdBu_r')
#cs1 = ax0.contourf(x, y, Bx,norm=colors.SymLogNorm(linthresh=0.1,linscale=0.1,vmin=Bx.min(), vmax=Bx.max()),cmap='RdBu_r')
ax3.set_title("Bx")
#ax0.set_xlabel("x ({})".format(units[unit_size]))
#ax0.set_ylabel("y ({})".format(units[unit_size]))
ax3.set_xlabel(r'$x$ (m)')
ax3.set_ylabel(r'$y$ (m)')
ax3.set_xlim(-XMAX/2, XMAX/2)
ax3.set_ylim(-YMAX/2, YMAX/2)
ax3.ticklabel_format(axis='both',style='sci',scilimits=(0,0),useMathText=True)
cbar4=fig2.colorbar(cs3,ax=ax3)
cbar4.set_label('Log(Bx)',rotation=270)
ax3.add_patch(Rectangle((-XM_Max,-YM_Max),length_mag,width_mag, color='k', zorder=100))

# =============================================================================
# Contour plot of By
# =============================================================================

ax4 = fig2.add_subplot(gs[0, 1])
norm4=colors.SymLogNorm(linthresh=0.000001,linscale=1, vmin=By2.min(), vmax=By2.max())
#cs2 = ax1.contourf(x, y, By, locator=ticker.LogLocator(),cmap='RdBu_r')
cs4 = ax4.contourf(x, y, By2[5], locator=ticker.SymmetricalLogLocator(base=10, linthresh=0.000001), norm=norm4,cmap='RdBu_r')
cbar5=fig2.colorbar(cs4, ax=ax4)
cbar5.set_label('Log(By)',rotation=270)
ax4.set_title("By")
#ax1.set_xlabel("x ({})".format(units[unit_size]))
#ax1.set_ylabel("y ({})".format(units[unit_size]))
ax4.set_xlabel(r'$x$ (m)')
ax4.set_ylabel(r'$y$ (m)')
ax4.set_xlim(-XMAX, XMAX)
ax4.set_ylim(-YMAX, YMAX)
ax4.ticklabel_format(axis='both',style='sci',scilimits=(0,0),useMathText=True)
ax4.add_patch(Rectangle((-XM_Max,-YM_Max),length_mag,width_mag, color='k', zorder=100))

# =============================================================================
# Contour plot of Bz
# =============================================================================

ax5 = fig2.add_subplot(gs[1, 0])
norm5=colors.SymLogNorm(linthresh=0.000001,linscale=1, vmin=Bz.min(), vmax=Bz.max())
#cs2 = ax1.contourf(x, y, By, locator=ticker.LogLocator(),cmap='RdBu_r')
cs5 = ax5.contourf(x, z, Bz[:,25,:], locator=ticker.SymmetricalLogLocator(base=10, linthresh=0.000001), norm=norm5,cmap='RdBu_r')
cbar6=fig2.colorbar(cs5, ax=ax5)
cbar6.set_label('Log(Bz)',rotation=270)
ax5.set_title("Bz")
#ax1.set_xlabel("x ({})".format(units[unit_size]))
#ax1.set_ylabel("y ({})".format(units[unit_size]))
ax5.set_xlabel(r'$x$ (m)')
ax5.set_ylabel(r'$z$ (m)')
ax5.set_xlim(-XMAX, XMAX)
ax5.set_ylim(-ZMAX, ZMAX)
ax5.ticklabel_format(axis='both',style='sci',scilimits=(0,0),useMathText=True)
ax5.add_patch(Rectangle((-XM_Max,-ZM_Max),length_mag,height_mag, color='k', zorder=100))

# =============================================================================
# Vector field plot
# =============================================================================
# Plot the streamlines with an appropriate colormap and arrow style

color2 = 2* np.log(np.hypot(Bx2[5], By2[5]))
ax6 = fig2.add_subplot(gs[1,1])
strm2=ax6.streamplot(x, y, Bx2[5], By2[5], color=color2, linewidth=1, cmap=plt.cm.inferno,density=1, arrowstyle='->', arrowsize=1.5)
cbar7=fig2.colorbar(strm2.lines,ax=ax6,extend='both')
cbar7.set_label(r'Log($\vert$B$\vert$)')
ax6.set_title("Vector field")
ax6.set_xlabel(r'$x$ (m)')
ax6.set_ylabel(r'$y$ (m)')
ax6.set_xlim(-XMAX, XMAX)
ax6.set_ylim(-YMAX, YMAX)
ax6.set_aspect('equal')
ax6.ticklabel_format(axis='both',style='sci',scilimits=(0,0),useMathText=True)
ax6.add_patch(Rectangle((-XM_Max,-YM_Max),length_mag,width_mag, color='k', zorder=100))
fig2.tight_layout()
plt.show()

# =============================================================================
# Vector field plot 3D
# =============================================================================
# Plot the streamlines with an appropriate colormap and arrow style

#fig3=plt.figure()
#ax7 = fig3.gca(projection='3d')
#ax7.quiver(X2,Y2,Z2,Bx2,By2,Bz, length=1,normalize=True)
#plt.show()