# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 09:08:54 2016

@author: herranz
"""

import warnings
import numpy                   as np
import matplotlib.pyplot       as plt
from   astropy.modeling        import models,fitting
from   astropy.modeling.models import custom_model
from   skimage.transform       import downscale_local_mean

from utils                     import img_shapefit,sigma2fwhm
from utils                     import makeGaussian

### ------ 'AJUSTE' CLASS:

class Ajuste:
    def __init__(self,model,amplitude,x,y,sigma,synthetic,gaussmap,residual):
        self.model = model
        self.amplitude = amplitude
        self.x = x
        self.y = y
        self.sigma = sigma
        self.synthetic = synthetic
        self.gaussmap = gaussmap
        self.residual = residual

def tiedfunc(g1):

    """
    Un ejemplo de parámetros 'tied'. En este caso, vamos a forzar a que el
    modelo gaussiano sea circularmente simétrico
    """
    y_stddev_1 = g1.x_stddev_1
    return y_stddev_1

### ------ SINGLE, ANALYTIC GAUSSIAN:

def fit_single_peak(patch,toplot=False,fixwidth=False,fixed_sigma=2.0,
                    fixcenter=False,center=None):

    """
       Fits an image patch to a composite model consisting of
    a planar baseline plus a symmetric Gaussian. Returns a fit object
    of the Ajuste class

    """

    m0 = patch.mean()
    a0 = patch.max()
    s0 = float(patch.shape[0])/8.0
    mm = np.where(patch==a0)
    a0 = a0 - m0
    if np.size(mm)==2:
        y0 = 0.5+float(mm[0])
        x0 = 0.5+float(mm[1])
    else:
        y0 = 0.5+float(mm[0][0])
        x0 = 0.5+float(mm[0][1])
    if fixcenter:
        if center is not None:
            x0 = center[0]
            y0 = center[1]
            a0 = patch.mean()
        else:
            x0 = patch.shape[0]//2 - 0.5
            y0 = patch.shape[1]//2 - 0.5
            a0 = patch.mean()

    y, x = np.mgrid[:patch.shape[0], :patch.shape[1]]


    model1 = models.Polynomial2D(degree=1)

    if fixwidth:
        if fixcenter:
            model2 = models.Gaussian2D(amplitude=a0,
                                       x_mean=x0,
                                       y_mean=y0,
                                       x_stddev=fixed_sigma,
                                       y_stddev=fixed_sigma,
                                       theta=0.0,
                                       fixed={'theta':True,
                                              'x_stddev':True,
                                              'x_mean':True,
                                              'y_mean':True})
        else:
            model2 = models.Gaussian2D(amplitude=a0,
                                       x_mean=x0,
                                       y_mean=y0,
                                       x_stddev=fixed_sigma,
                                       y_stddev=fixed_sigma,
                                       theta=0.0,
                                       fixed={'theta':True,
                                              'x_stddev':True})
    else:
        if fixcenter:
            model2 = models.Gaussian2D(amplitude=a0,
                                       x_mean=x0,
                                       y_mean=y0,
                                       x_stddev=s0,
                                       y_stddev=s0,
                                       theta=0.0,
                                       fixed={'theta':True,
                                              'x_mean':True,
                                              'y_mean':True})
        else:
            model2 = models.Gaussian2D(amplitude=a0,
                                       x_mean=x0,
                                       y_mean=y0,
                                       x_stddev=s0,
                                       y_stddev=s0,
                                       theta=0.0,
                                       fixed={'theta':True})

    modelo = model1+model2
    modelo.y_stddev_1.tied = tiedfunc

    fit_p  = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(modelo, x, y, patch)

    fit_amplitude = p.amplitude_1.value
    fit_x         = p.x_mean_1.value
    fit_y         = p.y_mean_1.value
    fit_sigma     = p.x_stddev_1.value

    sintetica     = p(x,y)
    residuo       = patch-sintetica
    modg          = models.Gaussian2D(amplitude=fit_amplitude,
                                      x_mean=fit_x,y_mean=fit_y,
                                      x_stddev=fit_sigma,y_stddev=fit_sigma,
                                      theta=0.0)
    gmap          = modg(x,y)

    f = Ajuste(p,fit_amplitude,fit_x,fit_y,fit_sigma,sintetica,gmap,residuo)

    if toplot:

        plt.figure(figsize=(12,12))
        plt.subplot(221)
        plt.pcolormesh(patch)
        plt.axis('tight')
        plt.title('Data')
        plt.colorbar()

        plt.subplot(222)
        plt.pcolormesh(sintetica)
        plt.axis('tight')
        plt.title('Planar baseline + Gaussian model')
        plt.colorbar()

        modg = models.Gaussian2D(amplitude=fit_amplitude,
                                 x_mean=fit_x,y_mean=fit_y,x_stddev=fit_sigma,
                                 y_stddev=fit_sigma,theta=0.0)
        plt.subplot(223)
        plt.pcolormesh(gmap)
        plt.axis('tight')
        plt.title('Gaussian model')
        plt.colorbar()

        plt.subplot(224)
        plt.pcolormesh(residuo)
        plt.axis('tight')
        plt.title('Residual')
        plt.colorbar()

    return f

### ------ PIXELIZED GAUSSIANS:

@custom_model
def pixelized_Gaussian2D(x,y,
                         resample_factor=4,
                         amplitude=1.0,
                         x_mean=1.0,
                         y_mean=1.0,
                         x_stddev=1.0,
                         y_stddev=1.0,
                         theta=0.0):

    x0, y0 = np.mgrid[:resample_factor[0]*x.shape[0], :resample_factor[0]*y.shape[0]]

    modg   = models.Gaussian2D(amplitude=amplitude,
                               x_mean=resample_factor[0]*x_mean[0],
                               y_mean=resample_factor[0]*y_mean[0],
                               x_stddev=resample_factor[0]*x_stddev[0],
                               y_stddev=resample_factor[0]*y_stddev[0],
                               theta=theta)

    image_big = modg(x0,y0)
    image     = downscale_local_mean(image_big,(resample_factor[0],resample_factor[0]))

    return image

def fit_single_pixelized_peak(patch,resample_factor=4,
                              toplot=False,fixwidth=False,fixed_sigma=2.0,
                              fixcenter=False,center=None):

    """
       Fits an image patch to a composite model consisting of
    a planar baseline plus a symmetric, pixelized Gaussian. Returns a fit object
    of the Ajuste class

    """

    m0 = patch.mean()
    a0 = patch.max()
    s0 = float(patch.shape[0])/8.0
    mm = np.where(patch==a0)
    a0 = a0 - m0
    if np.size(mm)==2:
        y0 = 0.5+float(mm[0])
        x0 = 0.5+float(mm[1])
    else:
        y0 = 0.5+float(mm[0][0])
        x0 = 0.5+float(mm[0][1])
    if fixcenter:
        if center is not None:
            x0 = center[0]
            y0 = center[1]
            a0 = patch.mean()
        else:
            x0 = patch.shape[0]//2 - 0.5
            y0 = patch.shape[1]//2 - 0.5
            a0 = patch.mean()

    y, x = np.mgrid[:patch.shape[0], :patch.shape[1]]


    model1 = models.Polynomial2D(degree=1)

    if fixwidth:
        if fixcenter:
            model2 = pixelized_Gaussian2D(resample_factor=resample_factor,
                                          amplitude=a0,
                                          x_mean=x0,
                                          y_mean=y0,
                                          x_stddev=fixed_sigma,
                                          y_stddev=fixed_sigma,
                                          theta=0.0,
                                          fixed={'resample_factor':True,
                                                 'theta':True,
                                                 'x_stddev':True,
                                                 'x_mean':True,
                                                 'y_mean':True})
        else:
            model2 = pixelized_Gaussian2D(resample_factor=resample_factor,
                                          amplitude=a0,
                                          x_mean=x0,
                                          y_mean=y0,
                                          x_stddev=fixed_sigma,
                                          y_stddev=fixed_sigma,
                                          theta=0.0,
                                          fixed={'resample_factor':True,
                                                 'theta':True,
                                                 'x_stddev':True})
    else:
        if fixcenter:
            model2 = pixelized_Gaussian2D(resample_factor=resample_factor,
                                          amplitude=a0,
                                          x_mean=x0,
                                          y_mean=y0,
                                          x_stddev=s0,
                                          y_stddev=s0,
                                          theta=0.0,
                                          fixed={'resample_factor':True,
                                                 'theta':True,
                                                 'x_mean':True,
                                                 'y_mean':True})
        else:
            model2 = pixelized_Gaussian2D(resample_factor=resample_factor,
                                          amplitude=a0,
                                          x_mean=x0,
                                          y_mean=y0,
                                          x_stddev=s0,
                                          y_stddev=s0,
                                          theta=0.0,
                                          fixed={'resample_factor':True,
                                                 'theta':True})

    modelo = model1+model2
    modelo.y_stddev_1.tied = tiedfunc

    fit_p  = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(modelo, x, y, patch)

    fit_amplitude = p.amplitude_1.value
    fit_x         = p.x_mean_1.value
    fit_y         = p.y_mean_1.value
    fit_sigma     = p.x_stddev_1.value

    sintetica     = p(x,y)
    residuo       = patch-sintetica
    modg          = pixelized_Gaussian2D(resample_factor=resample_factor,
                                         amplitude=fit_amplitude,
                                         x_mean=fit_x,y_mean=fit_y,
                                         x_stddev=fit_sigma,y_stddev=fit_sigma,
                                         theta=0.0)
    gmap          = modg(x,y)

    f = Ajuste(p,fit_amplitude,fit_x,fit_y,fit_sigma,sintetica,gmap,residuo)

    if toplot:

        plt.figure(figsize=(12,12))
        plt.subplot(221)
        plt.pcolormesh(patch)
        plt.axis('tight')
        plt.title('Data')
        plt.colorbar()

        plt.subplot(222)
        plt.pcolormesh(sintetica)
        plt.axis('tight')
        plt.title('Planar baseline + Gaussian model')
        plt.colorbar()

        modg = pixelized_Gaussian2D(resample_factor=resample_factor,
                                    amplitude=fit_amplitude,
                                    x_mean=fit_x,y_mean=fit_y,x_stddev=fit_sigma,
                                    y_stddev=fit_sigma,theta=0.0)
        plt.subplot(223)
        plt.pcolormesh(gmap)
        plt.axis('tight')
        plt.title('Gaussian model')
        plt.colorbar()

        plt.subplot(224)
        plt.pcolormesh(residuo)
        plt.axis('tight')
        plt.title('Residual')
        plt.colorbar()

    return f


### ------ FITS OF FILTERED GAUSSIANS:


filter_kernel = np.zeros((4,4),dtype=complex)
filter_kernel[2,2] += 1

@custom_model
def filtered_source_model(x,y,
                          resample_factor=4,
                          amplitude=1.0,
                          x_mean=1.0,
                          y_mean=1.0,
                          x_stddev=1.0,
                          y_stddev=1.0,
                          theta=0.0):

    image     = makeGaussian(x.shape[0],
                            fwhm=sigma2fwhm*x_stddev[0],
                            resample_factor=resample_factor[0],
                            center=(x_mean,y_mean))
    image     = image*amplitude
    l         = 2*int(sigma2fwhm*x_stddev[0])
    padscheme = ((l,l),(l,l))
    image     = np.pad(image,padscheme,mode='reflect')

    filtr_img = img_shapefit(filter_kernel.copy(),image)
    fimage    = np.fft.fftshift(np.fft.fft2(image))
    ffiltrada = np.multiply(fimage,filtr_img)
    image     = np.real(np.fft.ifft2(np.fft.ifftshift(ffiltrada)))
    image     = np.squeeze(image)
    size0     = x.shape[0]
    image     = image[l:l+size0,l:l+size0]

    return image

def fit_single_pixelized_filtered_peak(patch,
                                       filter_fftmap=None,
                                       resample_factor=8,
                                       toplot=False,
                                       fixwidth=True,
                                       fixed_sigma=2.0,
                                       fixcenter=False,
                                       center=None):

    """
       Fits an image patch to a composite model consisting of
    a planar baseline plus a symmetric, pixelized Gaussian filtered by the
    filter FILTER_FFTMAP. Returns a fit object of the Ajuste class

    """

    global filter_kernel

    filter_kernel = filter_fftmap

    m0 = patch.mean()
    a0 = patch.max()
    mm = np.where(patch==a0)
    a0 = a0 - m0
    if np.size(mm)==2:
        x0 = 0.5+float(mm[0])
        y0 = 0.5+float(mm[1])
    else:
        x0 = 0.5+float(mm[0][0])
        y0 = 0.5+float(mm[0][1])
    if fixcenter:
        if center is not None:
            x0 = center[0]
            y0 = center[1]
            a0 = patch.mean()
        else:
            x0 = patch.shape[0]//2 - 0.5
            y0 = patch.shape[1]//2 - 0.5
            a0 = patch.mean()

    x, y = np.mgrid[:patch.shape[0], :patch.shape[1]]

    model1 = models.Polynomial2D(degree=1)

    if fixwidth:
        if fixcenter:
            model2 = filtered_source_model(resample_factor=resample_factor,
                                           amplitude=a0,
                                           x_mean=x0,
                                           y_mean=y0,
                                           x_stddev=fixed_sigma,
                                           y_stddev=fixed_sigma,
                                           theta=0.0,
                                           fixed={'resample_factor':True,
                                                  'theta':True,
                                                  'x_stddev':True,
                                                  'x_mean':True,
                                                  'y_mean':True})
        else:
            model2 = filtered_source_model(resample_factor=resample_factor,
                                           amplitude=a0,
                                           x_mean=x0,
                                           y_mean=y0,
                                           x_stddev=fixed_sigma,
                                           y_stddev=fixed_sigma,
                                           theta=0.0,
                                           fixed={'resample_factor':True,
                                                  'theta':True,
                                                  'x_stddev':True})
    else:
        if fixcenter:
            model2 = filtered_source_model(resample_factor=resample_factor,
                                           amplitude=a0,
                                           x_mean=x0,
                                           y_mean=y0,
                                           x_stddev=fixed_sigma,
                                           y_stddev=fixed_sigma,
                                           theta=0.0,
                                           fixed={'resample_factor':True,
                                                  'theta':True,
                                                  'x_mean':True,
                                                  'y_mean':True})
        else:
            model2 = filtered_source_model(resample_factor=resample_factor,
                                           amplitude=a0,
                                           x_mean=x0,
                                           y_mean=y0,
                                           x_stddev=fixed_sigma,
                                           y_stddev=fixed_sigma,
                                           theta=0.0,
                                           fixed={'resample_factor':True,
                                                  'theta':True})

    modelo = model1+model2
    modelo.y_stddev_1.tied = tiedfunc

    fit_p  = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(modelo, x, y, patch)

    fit_amplitude = p.amplitude_1.value
    fit_x         = p.x_mean_1.value
    fit_y         = p.y_mean_1.value
    fit_sigma     = p.x_stddev_1.value

    sintetica     = p(x,y)
    residuo       = patch-sintetica
    modg          = filtered_source_model(resample_factor=resample_factor,
                                          amplitude=fit_amplitude,
                                          x_mean=fit_x,
                                          y_mean=fit_y,
                                          x_stddev=fit_sigma,
                                          y_stddev=fit_sigma,
                                          theta=0.0)
    gmap          = modg(x,y)

    f = Ajuste(p,fit_amplitude,fit_x,fit_y,fit_sigma,sintetica,gmap,residuo)

    if toplot:

        plt.figure(figsize=(12,12))
        plt.subplot(221)
        plt.pcolormesh(patch)
        plt.axis('tight')
        plt.title('Data')
        plt.colorbar()

        plt.subplot(222)
        plt.pcolormesh(sintetica)
        plt.axis('tight')
        plt.title('Planar baseline + source model')
        plt.colorbar()


        plt.subplot(223)
        plt.pcolormesh(gmap)
        plt.axis('tight')
        plt.title('source model')
        plt.colorbar()

        plt.subplot(224)
        plt.pcolormesh(residuo)
        plt.axis('tight')
        plt.title('Residual')
        plt.colorbar()

    return f