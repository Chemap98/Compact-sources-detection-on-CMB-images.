#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:33:46 2020

      MISCELLANEOUS UTILITIES FOR THE DETECTION PACKAGES
    
@author: herranz
"""

### ---- GENERAL PURPOSE IMPORTS:

import numpy as  np
import healpy as hp
import matplotlib.pyplot as plt
import time

from scipy             import ndimage
from scipy.interpolate import interp1d

### ---- BASIC UNIT TRANSFORMATIONS:

fwhm2sigma   = 1.0/(2.0*np.sqrt(2.0*np.log(2.0)))
sigma2fwhm   = 1.0/fwhm2sigma

### ---- FAST ARITHMETICAL COMPUTATIONS USING NUMBA:

import numba

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

### ---- MATRIX CROPPING / PADDING

def img_shapefit(img1,img2):

    """
    Forces image1 to have the same shape as image2. If image1 was larger than
    image2, then it is cropped in its central part. If image1 was smaller that
    image2, then it is padded with zeros. Dimensions must have even size
    """

    if img1.ndim == 2:
        (n1,n2) = img1.shape
    else:
        n1 = np.sqrt(img1.size)
        n2 = np.sqrt(img1.size)
    (m1,m2) = img2.shape
    (c1,c2) = (n1//2,n2//2)
    (z1,z2) = (m1//2,m2//2)

    img3 = np.zeros((m1,m2),dtype=img1.dtype)

    if n1<=m1:
        if n2<=m2:
            img3[z1-c1:z1+c1,z2-c2:z2+c2] = img1        # Standard padding
        else:
            img3[z1-c1:z1+c1,:] = img1[:,c2-z2:c2+z2]
    else:
        if n2<=m2:
            img3[:,z2-c2:c2+z2] = img1[c1-z1:c1+z1,:]
        else:
            img3 = img1[c1-z1:c1+z1,c2-z2:c2+z2]

    return img3


### ---- FAST COMPUTATION OF TWO-DIMENSIONAL RADIALLY SYMMETRIC PROFILES:
    
from skimage.transform import downscale_local_mean

def makeGaussian(size,fwhm=3,
                 resample_factor=1,
                 center=None,
                 verbose=False,
                 toplot=False):

    """

    Makes a square image of a Gaussian kernel, returned as a numpy
    two-dimensional array. The Gaussian takes a maximum value = 1
    at the position defined in 'center'

    PARAMETERS:
        'size'    is the length of a side of the square
        'fwhm'    is full-width-half-maximum, in pixel units, of the
                     Gaussian kernel.
        'resample_factor' indicates how to increase (or not, if equal
                     to one) the image
        'center'  is a tuple,list or numpy array containing the (x,y)
                     coordinates (in pixel units) of the centre of the
                     Gaussian kernel. If certer=None, the Gaussian is
                     placed at the geometrical centre of the image
        'verbose' if true, the routine writes out the info about the
                     code
        'toplot'  if true, plots the output array

    """

    start_time = time.time()

    if center is None:
        x0 = y0 = resample_factor*size // 2
    else:
        x0 = resample_factor*center[0]
        y0 = resample_factor*center[1]

    if verbose:
        print(' ')
        print(' --- Generating a {0}x{0} image with a Gaussian kernel of FWHM = {1} pixels located at ({2},{3})'.format(size,fwhm,x0/resample_factor,y0/resample_factor))

    y = np.arange(0, resample_factor*size, 1, float)
    x = y[:,np.newaxis]                # This couple of lines generates a very efficient
                                       # structure of axes (horizontal and vertical)
                                       # that can be filled very quickly with a function
                                       # such as the Gaussian defined in the next line.
                                       # This is far much faster than the old-fashioned
                                       # nested FOR loops.

    u = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / (resample_factor*fwhm)**2)

    if resample_factor != 1:
        u = downscale_local_mean(u,(resample_factor,resample_factor))

    if toplot:
        plt.figure()
        plt.imshow(u)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')

    if verbose:
        print(' --- Gaussian template generated in {0} seconds'.format(time.time() - start_time))

    return u



def radial_profile(data,nbins=50,center=None,
                   toplot=False,kind='linear',datype='real',
                   equal_scale=False):

    """

    Given an image in the array DATA, this routine creates a bin-averaged
    radial profile form point CENTER (the center of the image, if no CENTER
    is provided)

    """    
    
    
    s       = data.shape
    dmean   = data.mean()
    dstd    = data.std()
    size    = s[0]
    perfil  = [None] * nbins
    xperfil =  [None] * nbins

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    r = np.sqrt((x-x0)**2+(y-y0)**2)

    rbin = (nbins * r/r.max()).astype(np.int)

    if datype=='real':
        perfil  = ndimage.mean(data, labels=rbin, index=np.arange(0, rbin.max() ))
    else:
        perfilr = ndimage.mean(np.real(data), labels=rbin, index=np.arange(0, rbin.max() ))
        perfili = ndimage.mean(np.imag(data), labels=rbin, index=np.arange(0, rbin.max() ))
        perfil  = perfilr+np.complex(0,1)*perfili

    xperfil = [i for i in range(nbins)]
    xperfil = np.multiply(xperfil,r.max())
    xperfil = np.divide(xperfil,float(nbins))+r.max()/(2.0*nbins)

    perfil  = [data[x0,y0]] + np.array(perfil).tolist()
    xperfil = [0] + np.array(xperfil).tolist()

    if datype=='real':
        f = interp1d(xperfil, perfil, kind=kind,bounds_error=False,fill_value=perfil[nbins-1])
        if kind=='linear':
            pmap = np.interp(r,xperfil,perfil)
        else:
            pmap = f(r)
    else:
        f1 = interp1d(xperfil, np.real(perfil), kind=kind,bounds_error=False,fill_value=np.real(perfil[nbins-1]))
        f2 = interp1d(xperfil, np.imag(perfil), kind=kind,bounds_error=False,fill_value=np.imag(perfil[nbins-1]))
        if kind=='linear':
            pmap = np.interp(r,xperfil,np.real(perfil))+np.complex(0,1)*np.interp(r,xperfil,np.imag(perfil))
        else:
            pmap = f1(r) + np.complex(0,1)*f2(r)

    if toplot:
        plt.figure()
        xnew = np.linspace(0,np.max(xperfil),1001,endpoint=True)
        if datype=='real':
            plt.plot(xnew,f(xnew))
        else:
            plt.plot(xnew,f1(xnew))
        plt.show()
        plt.figure()
        plt.pcolormesh(pmap)
        plt.axis('tight')
        plt.colorbar()
        plt.show()

    if equal_scale:
        pmap = dstd*pmap/pmap.std()
        pmap = pmap-pmap.mean()+dmean

    return pmap,xperfil,perfil


### ---- COORDINATE UTILITIES:
    
def coord2vec(coord,coordsys='G'):
    if coordsys.upper()[0] == 'G':
        lon = coord.galactic.l.deg
        lat = coord.galactic.b.deg
    else:
        lon = coord.icrs.ra.deg
        lat = coord.icrs.dec.deg
    vec = hp.ang2vec(lon,lat,lonlat=True)
    return vec


