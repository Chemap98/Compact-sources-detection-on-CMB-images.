# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:43:34 2016

    Utils for computing elementary statistics on 2D arrays (images):
        mean, stddev, median, sum, pixel counts, etc
        in rings and circles around given coordinates
        (the center of the image is used as a default)


@author: herranz
"""
import numpy         as np
import scipy.ndimage as nd
import astropy.units as u
from   scipy.stats   import sigmaclip


### ------------------------------------------------------------------
###    RADIAL AND ANGULAR IMAGES
### ------------------------------------------------------------------

"""
      Here we define routines for fast generation of radial and angular
      maps (i.e. images that contain the distance to a given point and angles
      around that point)
"""

def xy_arrays(size,center=None):
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    y = np.arange(0,size,1,float)
    x = y[:,np.newaxis]
    
    x = x-x0
    y = y-y0
    
    return x,y

def distance_map(size,center=None,units=None):
    
    x,y = xy_arrays(size,center=center)
    r   = np.sqrt(x**2+y**2)   
    if units is not None:
        r = r*units
    
    return r

def angle_map(size,center=None,deg=False,phase_rad=None):
    
    x,y = xy_arrays(size,center=center)
    phi = np.arctan2(x,y)
    
    if phase_rad is not None:
        phi += phase_rad
    
    if deg:
        phi = phi*(u.rad.to(u.deg))
        
    return phi
    

### ------------------------------------------------------------------
###    STATISTICS IN RINGS AND CIRCLES
### ------------------------------------------------------------------

def ring_mask(n,r1,r2,center=None):
    x = np.arange(0, n, 1, float)
    y = x[:,np.newaxis]
    if center is None:
        c1 = c2 = n // 2
    else:
        c1,c2 = center
    r = np.sqrt((x-c1)**2+(y-c1)**2)
    m = np.where((r>=r1)&(r<=r2),1,0)
    return m

def ring_mean(imagen,r1,r2,centro=None,clip=None):
    labels = ring_mask(imagen.shape[0],r1,r2,center=centro)
    datos  = imagen[labels==1].flatten()
    if clip is not None:
        datos,lower,upper = sigmaclip(datos,low=clip,high=clip)
    return datos.mean()

def ring_min(imagen,r1,r2,centro=None,clip=None):
    labels = ring_mask(imagen.shape[0],r1,r2,center=centro)
    datos  = imagen[labels==1].flatten()
    if clip is not None:
        datos,lower,upper = sigmaclip(datos,low=clip,high=clip)
    return datos.min()

def ring_max(imagen,r1,r2,centro=None,clip=None):
    labels = ring_mask(imagen.shape[0],r1,r2,center=centro)
    datos  = imagen[labels==1].flatten()
    if clip is not None:
        datos,lower,upper = sigmaclip(datos,low=clip,high=clip)
    return datos.max()

def ring_std(imagen,r1,r2,centro=None,clip=None):
    labels = ring_mask(imagen.shape[0],r1,r2,center=centro)
    datos  = imagen[labels==1].flatten()
    if clip is not None:
        datos,lower,upper = sigmaclip(datos,low=clip,high=clip)
    return datos.std()

def max_in_circle(imagen,r,centro=None):
    labels = ring_mask(imagen.shape[0],0,r,center=centro)
    return nd.maximum(imagen,labels,1)

def min_in_circle(imagen,r,centro=None):
    labels = ring_mask(imagen.shape[0],0,r,center=centro)
    return nd.minimum(imagen,labels,1)

def sum_in_circle(imagen,r,centro=None):
    labels = ring_mask(imagen.shape[0],0,r,center=centro)
    return nd.sum(imagen,labels,1)

def std_in_circle(imagen,r,centro=None):
    labels = ring_mask(imagen.shape[0],0,r,center=centro)
    return nd.standard_deviation(imagen,labels,1)

def ring_sum(imagen,r1,r2,centro=None,clip=None):
    labels = ring_mask(imagen.shape[0],r1,r2,center=centro)
    datos  = imagen[labels==1].flatten()
    if clip is not None:
        datos,lower,upper = sigmaclip(datos,low=clip,high=clip)
    return datos.sum()

def ring_count(imagen,r1,r2,centro=None,clip=None):
    labels = ring_mask(imagen.shape[0],r1,r2,center=centro)
    datos  = imagen[labels==1].flatten()
    if clip is not None:
        datos,lower,upper = sigmaclip(datos,low=clip,high=clip)
    return datos.size

def count_in_circle(imagen,r,centro=None):
    labels = ring_mask(imagen.shape[0],0,r,center=centro)
    return nd.sum(np.zeros(imagen.shape),labels,1)

def median_in_circle(imagen,r,centro=None):
    labels = ring_mask(imagen.shape[0],0,r,center=centro)
    return nd.median(imagen,labels,1)

def ring_median(imagen,r1,r2,centro=None,clip=None):
    labels = ring_mask(imagen.shape[0],r1,r2,center=centro)
    datos  = imagen[labels==1].flatten()
    if clip is not None:
        datos,lower,upper = sigmaclip(datos,low=clip,high=clip)
    return np.median(datos)
