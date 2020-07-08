# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 21:15:49 2016

   This program filters an image stored in the array DATA
   and filters it with a matched fiter for a profile template
   TPROF or a Gaussian shaped template profile with full width
   at half maximum FWHM (in pixel units). It has been tested to
   give a similar performance to the Fortran90
   implementation by D. Herranz. Running time has not been compared.

   Optionally, images can be padded (with a reflection of the
   data inside) in order to reduce border effects, with a cost
   in running time. This option is activated by default.

@author: herranz
"""
import time
import numpy             as np
import matplotlib.pyplot as plt
from   skimage.feature   import peak_local_max

import gauss2dfit        as gfit
from   utils             import makeGaussian
from   utils             import radial_profile
from   utils             import abs2

def matched_filter(data0,gprof=True,
                   lafwhm=2.0,tprof0=None,
                   resample_factor=1,
                   nbins=50,aplot=False,
                   topad=True,kind='linear',
                   return_filtered_profile=False,
                   verbose=False):

    start_time = time.time()

    s = data0.shape
    size0 = s[0]

    if verbose:
        print("--- Image size %d x %d pixels ---" % (size0,size0))

    if topad:
        l    = 2*int(lafwhm)
        padscheme = ((l,l),(l,l))
        data = np.pad(data0,padscheme,mode='reflect')
        if tprof0 is not None:
            tprof = np.pad(tprof0,padscheme,mode='reflect')
    else:
        l = 0
        data = np.copy(data0)

    s = data.shape
    size = s[0]

    if gprof:
        g  = makeGaussian(size,fwhm=lafwhm,
                          resample_factor=resample_factor)
    else:
        if not topad:
            tprof = tprof0.copy()
        g = tprof

    gf = np.abs(np.fft.fftshift(np.fft.fft2(g)))

    fdata = np.fft.fftshift(np.fft.fft2(data))
    P     = abs2(fdata)

    profmap,xp,yp = radial_profile(P,nbins,
                                   toplot=aplot,
                                   kind=kind,
                                   datype='real')

    filtro        = np.divide(gf,profmap)
    pc            = size//2
    filtro[pc,pc] = 0.0*filtro[pc,pc]

    fparanorm = np.multiply(gf,filtro)
    paranorm = np.real(np.fft.ifft2(np.fft.ifftshift(fparanorm)))
    normal = np.amax(paranorm)/np.amax(g)
    filtro = filtro/normal

    ffiltrada = np.multiply(fdata,filtro)

    filtrada_pad = np.real(np.fft.ifft2(np.fft.ifftshift(ffiltrada)))

    filtrada = filtrada_pad[l:l+size0,l:l+size0]

    if aplot:
        plt.figure()
        plt.pcolormesh(data[l:l+size0,l:l+size0])
        plt.axis('tight')
        plt.title('Original data')
        plt.colorbar()

        plt.figure()
        plt.pcolormesh(filtrada)
        plt.axis('tight')
        plt.title('Filtered data')
        plt.colorbar()

    if verbose:
        print("--- Matched filtered in %s seconds ---" % (time.time() - start_time))

    if return_filtered_profile:
        filtrada    = filtro
                # it is returned in Fourier space with padding

    return filtrada


def iterative_matched_filter(data0,lafwhm=2.0,
                             resample_factor=1,
                             nbins=50,snrcut=5.0,
                             aplot=False,topad=True,kind='linear',
                             return_filtered_profile=False,
                             verbose=False):
    """
    Esta rutina asume que el perfil de las fuentes es gaussiano y que
    se conoce la anchura de las fuentes. Solo funciona en ese caso. La
    rutina hace un primer filtrado de los datos, identifica picos en
    la imagen filtrada con SNR mayor que SNRCUT y los elimina de la
    imagen mediante ajustes consecutivos a los datos en pequeños parches
    alrededor de los picos, para obtener un "background" limpio con el
    cual re-calcular el matched filter. Se da como salida los dos mapas
    filtrados: aquel en el que no se ha hecho proceso iterativo, y el
    mapa filtrado con el filtro iterado.

    """

    start_time = time.time()

    lasigma = lafwhm/(2.0*np.sqrt(2.0*np.log(2.0)))

    s = data0.shape
    size0 = s[0]

    if verbose:
        print("--- Image size %d x %d pixels ---" % (size0,size0))

    if topad:
        l    = 2*int(lafwhm)
        padscheme = ((l,l),(l,l))
        data = np.pad(data0,padscheme,mode='reflect')
    else:
        l = 0
        data = np.copy(data0)

    s    = data.shape
    size = s[0]
    pc   = size//2

#   Gaussian profile (in real and Fourier space):

    g  = makeGaussian(size,fwhm=lafwhm,resample_factor=resample_factor)
    gf = np.abs(np.fft.fftshift(np.fft.fft2(g)))

#   FFT and power spectrum map of the original data:

    fdata            = np.fft.fftshift(np.fft.fft2(data))
    P0               = np.abs(fdata)**2
    profmap0,xp0,yp0 = radial_profile(P0,nbins,toplot=aplot,kind=kind)
#    profmap0[profmap0==0] = 1

#   First iteration of the normalised filter:

    filtro1        = np.divide(gf,profmap0)
    filtro1[pc,pc] = 0.0
    fparanorm      = np.multiply(gf,filtro1)
    paranorm       = np.real(np.fft.ifft2(np.fft.ifftshift(fparanorm)))
    normal         = np.amax(paranorm)/np.amax(g)
    filtro1        = filtro1/normal

#   First filtering:

    ffiltrada1     = np.multiply(fdata,filtro1)
    filtrada1_pad  = np.real(np.fft.ifft2(np.fft.ifftshift(ffiltrada1)))
    filtrada1      = filtrada1_pad[l:l+size0,l:l+size0]

#   Detection of the peaks above a cut in SNR,
#    fit to a Gaussian around that regions and substraction from the
#    input map

    picos  = peak_local_max(filtrada1,min_distance=int(lafwhm),
                            threshold_abs=snrcut*filtrada1.std(),
                            exclude_border=int(lafwhm),
                            indices=True)
    npicos = len(picos)

    if npicos<1:

        filtrada   = filtrada1  # It isn't necessary to iterate
        if verbose:
            print(' ')
            print(' ---- No peaks above threshold in the filtered image ')
            print(' ')
        f = 0

    else:

        data1 = np.copy(data0)

        if verbose:
            print(' ')
            print(' ---- {0} peaks above threshold '.format(npicos))
            print(' ')

        for pico in range(npicos):

#   We select a patch around that peak

            lugar = picos[pico,:]

            nx   = int(2.5*lafwhm)
            xinf = int(lugar[0])-nx
            if xinf<0:
                xinf = 0
            xsup = int(lugar[0])+nx
            if xsup>(size0-1):
                xsup = size0-1
            ny   = int(2.5*lafwhm)
            yinf = int(lugar[1])-ny
            if yinf<0:
                yinf = 0
            ysup = int(lugar[1])+ny
            if ysup>(size0-1):
                ysup = size0-1

#   We fit to a Gaussian profile with fixed width in that patch

            patch = data1[xinf:xsup,yinf:ysup]
            if resample_factor == 1:
                f = gfit.fit_single_peak(patch,toplot=aplot,
                                         fixwidth=True,fixed_sigma=lasigma)
            else:
                f = gfit.fit_single_pixelized_peak(patch,
                                                   resample_factor=resample_factor,
                                                   toplot=aplot,fixwidth=True,
                                                   fixed_sigma=lasigma)

#   We subtract the fitted Gaussian from a copy of the original data

            data1[xinf:xsup,yinf:ysup] = data1[xinf:xsup,yinf:ysup] - f.gaussmap

#   Plot the cleaned map:

        if aplot:
            plt.figure()
            plt.pcolormesh(data1)
            plt.axis('tight')
            plt.colorbar()
            plt.title('Original data with brightests sources removed')

#   Second interation of the filter:

        if topad:
            l    = 2*int(lafwhm)
            padscheme = ((l,l),(l,l))
            data2 = np.pad(data1,padscheme,mode='reflect')
        else:
            l = 0
            data2 = data1

#   FFT and power spectrum map of the original data:

        fdata2           = np.fft.fftshift(np.fft.fft2(data2))
        P1               = np.abs(fdata2)**2
        profmap1,xp1,yp1 = radial_profile(P1,nbins,toplot=aplot,kind=kind)

#   Second iteration of the normalised filter:

        filtro2        = np.divide(gf,profmap1)
        filtro2[pc,pc] = 0.0
        fparanorm      = np.multiply(gf,filtro2)
        paranorm       = np.real(np.fft.ifft2(np.fft.ifftshift(fparanorm)))
        normal         = np.amax(paranorm)/np.amax(g)
        filtro2        = filtro2/normal

#   Second filtering:

        ffiltrada2     = np.multiply(fdata,filtro2)
        filtrada2_pad  = np.real(np.fft.ifft2(np.fft.ifftshift(ffiltrada2)))
        filtrada       = filtrada2_pad[l:l+size0,l:l+size0]


    if aplot:
        plt.figure()
        plt.pcolormesh(data[l:l+size0,l:l+size0])
        plt.axis('tight')
        plt.title('Original data')
        plt.colorbar()

        plt.figure()
        plt.pcolormesh(filtrada)
        plt.axis('tight')
        plt.title('Filtered data')
        plt.colorbar()

    if verbose:
        print("--- Matched filtered in %s seconds ---" % (time.time() - start_time))

    if return_filtered_profile:
        filtrada = paranorm/np.amax(paranorm)

    return filtrada1,filtrada



