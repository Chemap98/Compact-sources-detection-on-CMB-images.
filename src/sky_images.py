#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:43:56 2018

@author: herranz
"""

import numpy as np
import matplotlib.pyplot as plt
import matched_filter as mf
import healpy as hp
import astropy.units as u
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.nddata import block_reduce,block_replicate
from astropy.nddata import Cutout2D
from scipy.stats import describe
from image_utils import ring_min,ring_max,ring_mean,ring_std,ring_sum,ring_count,ring_median
from image_utils import min_in_circle,max_in_circle,sum_in_circle,count_in_circle,median_in_circle
from utils import makeGaussian
from utils import coord2vec,img_shapefit,sigma2fwhm,fwhm2sigma
from gauss2dfit import fit_single_peak
from scipy.ndimage import gaussian_filter

class Imagen:

    image_header   = None
    image_coordsys = 'galactic'

    def __init__(self,datos,centro,size,pixsize):
        self.datos   = datos
        self.centro  = centro
        self.size    = size
        self.pixsize = pixsize

    def __add__(self,other):
        if isinstance(other, (int, float, complex)):
            datos = self.datos+other
        else:
            datos = self.datos+other.datos
        return Imagen(datos,self.centro,self.size,self.pixsize)

    def __radd__(self, other):                         # reverse SUM
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self,other):
        if isinstance(other, (int, float, complex)):
            datos = self.datos-other
        else:
            datos = self.datos-other.datos
        datos = self.datos-other
        return Imagen(datos,self.centro,self.size,self.pixsize)

    def __rsub__(self,other):                         # reverse SUM
        if other == 0:
            return self
        else:
            return self.__sub__(other)

    def __mul__(self,other):
        if isinstance(other, (int, float, complex)):
            datos = self.datos*other
        else:
            datos = self.datos*other.datos
        return Imagen(datos,self.centro,self.size,self.pixsize)

    def __rmul__(self, other):                         # reverse SUM
        if other == 0:
            return self
        else:
            return self.__mul__(other)

    def copy(self):
        return Imagen(self.datos,self.centro,self.size,self.pixsize)

# ------- BASIC DESCRIPTION AND STATISTICS -------------

    def std(self):
        return self.datos.std()

    def mean(self):
        return self.datos.mean()

    def minmax(self):
        return self.datos.min(),self.datos.max()

    def stats_in_rings(self,inner,outer,clip=None):
        rmin = (inner/self.pixsize).si.value
        rmax = (outer/self.pixsize).si.value
        img  = self.datos
        if rmin > 0.0:
            s = ring_sum(img,rmin,rmax)
            n = ring_count(img,rmin,rmax)
            m = ring_median(img,rmin,rmax)
        else:
            s = sum_in_circle(img,rmax)
            n = count_in_circle(img,rmax)
            m = median_in_circle(img,rmax)

        return {'min_inside':min_in_circle(img,rmin),
                'max_inside':max_in_circle(img,rmin),
                'min_ring':ring_min(img,rmin,rmax,clip=clip),
                'max_ring':ring_max(img,rmin,rmax,clip=clip),
                'mean':ring_mean(img,rmin,rmax,clip=clip),
                'std':ring_std(img,rmin,rmax,clip=clip),
                'sum':s,
                'count':n,
                'median':m}

    @property
    def stats(self):
        return describe(self.datos,axis=None)

    @property
    def pixsize_deg(self):
        return self.pixsize.to(u.deg).value

    @property
    def lsize(self):
        return self.datos.shape[0]

# ------- INPUT/OUTPUT --------------------------------

    def write(self,fname):
        exts_fits  = ['fits','fits.gz']
        exts_ascii = ['txt','dat']
        exts_img   = ['eps','jpeg','jpg','pdf','pgf','png',
                      'ps','raw','rgba','svg','svgz','tif','tiff']
        extension = fname.split('.')[-1].lower()
        if extension in exts_fits:
            hdu = fits.PrimaryHDU(np.fliplr(np.flipud(self.datos)))
            h   = self.wcs.to_header()
            hdu.header += h
            hdu.writeto(fname,overwrite=True)
        elif extension in exts_ascii:
            n = self.lsize
            with open(fname, 'w') as f:
                f.write('# GLON: {0}\n'.format(self.center_coordinate.galactic.l.deg))
                f.write('# GLAT: {0}\n'.format(self.center_coordinate.galactic.b.deg))
                f.write('# PIXSIZE: {0}\n'.format(self.pixsize_deg))
                for i in range(n):
                    for j in range(n):
                        f.write('{0}  {1}  {2}\n'.format(i+1,j+1,self.datos[i,j]))
        elif extension in exts_img:
            fig = plt.figure()
            self.draw(newfig=False)
            fig.savefig(fname)
            plt.close()
        else:
            print(' --- Unknown file type')

    @classmethod
    def from_ascii_file(self,fname):
        with open(fname) as f:
            lines = f.readlines()
        lsize  = int(np.sqrt(len(lines)-3))
        datos  = np.zeros((lsize,lsize))
        glon   = float((lines[0].split('GLON: ')[-1]).split('\n')[0])
        glat   = float((lines[1].split('GLAT: ')[-1]).split('\n')[0])
        psiz   = float((lines[2].split('PIXSIZE: ')[-1]).split('\n')[0])
        centro = np.array([90.0-glat,glon])
        size   = (lsize,lsize)
        pixs   = psiz*u.deg
        for k in range(3,len(lines)):
            l = lines[k].split(' ')
            i = int(l[0])-1
            j = int(l[2])-1
            z = float((l[4]).split('\n')[0])
            datos[i,j] = z
        return Imagen(datos,centro,size,pixs)

    @classmethod
    def from_fits_file(self,fname):
        hdul = fits.open(fname)
        imag = self.from_hdu(hdul[0])
        hdul.close()
        return imag

    @classmethod
    def from_hdu(self,hdul):
        h    = hdul.header
        w    = wcs.WCS(h)
        co   = w.pixel_to_world(h['NAXIS1']/2,h['NAXIS2']/2)

        try:
            pixsize = u.Quantity(np.abs(h['CDELT1']),unit=h['CUNIT1'])
        except KeyError:
            try:
                pixsize = u.Quantity(np.abs(h['CDELT1']),unit=u.deg)
            except KeyError:
                pixsize = np.abs(w.pixel_scale_matrix).mean()*u.deg

        centro  = np.array([90.0-co.galactic.b.deg,co.galactic.l.deg])
        datos   = np.fliplr(np.flipud(hdul.data))
        size    = hdul.data.shape

        self.image_header = h
        if (('RA' in h['CTYPE1'].upper()) or ('RA' in h['CTYPE2'].upper())):
            self.image_coordsys = 'icrs'
        else:
            self.image_coordsys = 'galactic'

        return Imagen(datos,centro,size,pixsize)

    @classmethod
    def from_file(self,fname):
        exts_fits  = ['fits']
        exts_ascii = ['txt','dat']
        extension = fname.split('.')[-1].lower()
        if extension in exts_fits:
            newimg = self.from_fits_file(fname)
        elif extension in exts_ascii:
            newimg = self.from_ascii_file(fname)
        else:
            print(' --- Unknown file type')
            newimg = []
        return newimg

    @classmethod
    def empty(self,npix,pixsize):
        d = np.zeros((npix,npix))
        return Imagen(d,(45.0,45.0),(npix,npix),pixsize)


# ------- PLOTTING ----------------------

    def plot(self):
        wcs = self.wcs
        fig = plt.figure()
        fig.add_subplot(111, projection=wcs)
        plt.imshow(np.flipud(np.fliplr(self.datos)),  cmap=plt.cm.viridis)
        plt.xlabel('RA')
        plt.ylabel('Dec')

    def draw(self,newfig=False,animated=False,tofile=None):
        g = np.arange(self.size[0])-float(self.size[0])/2
        g = (g+0.5)*self.pixsize_deg
        if self.image_coordsys == 'galactic':
            x = g+self.center_coordinate.galactic.l.deg
            y = g+self.center_coordinate.galactic.b.deg
        else:
            x = g+self.center_coordinate.icrs.ra.deg
            y = g+self.center_coordinate.icrs.dec.deg

        X, Y = np.meshgrid(x, y)
        if newfig:
            plt.figure()
        plt.pcolormesh(X,Y,np.flipud(np.fliplr(self.datos)),animated=animated)
        plt.axis('tight')
        if self.image_coordsys == 'galactic':
            plt.ylabel('GLAT [deg]')
            plt.xlabel('GLON [deg]')
        else:
            plt.ylabel('RA [deg]')
            plt.xlabel('DEC [deg]')
        plt.colorbar()
        if tofile is not None:
            plt.savefig(tofile)


# ------- HEADER --------------------

    @property
    def header(self):
        if self.image_header is None:
            w      = self.wcs
            hdu    = fits.PrimaryHDU(self.datos)
            hdu.header.update(w.to_header())
            self.image_header = hdu.header
        return self.image_header


# ------- COORDINATES  -------------

    @property
    def wcs(self):

        if self.image_header is not None:

            w = wcs.WCS(self.image_header)

        else:

            n = self.size[0]//2
            c = self.center_coordinate
            w = wcs.WCS(naxis=2)
            w.wcs.crpix = [n,n]
            w.wcs.cdelt = np.array([self.pixsize_deg,self.pixsize_deg])
            if self.image_coordsys == 'galactic':
                w.wcs.crval = [c.galactic.l.deg, c.galactic.b.deg]
                w.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
            else:
                w.wcs.crval = [c.icrs.ra.deg, c.icrs.dec.deg]
                w.wcs.ctype = ["RA-TAN", "DEC-TAN"]
            w.wcs.cunit = ['deg','deg']

        return w

    @property
    def center_coordinate(self):
        return SkyCoord(frame='galactic',
                        b=(90.0-self.centro[0])*u.deg,
                        l=self.centro[1]*u.deg)

    def pixel_coordinate(self,i,j):
        s = self.size
        p = np.array(self.wcs.wcs_pix2world(s[1]-j,s[0]-i,1))
        return SkyCoord(p[0],p[1],unit='deg',frame=self.image_coordsys)

    def coordinate_pixel(self,coord):
        if self.image_coordsys == 'galactic':
            l = coord.galactic.l.deg
            b = coord.galactic.b.deg
        else:
            l = coord.icrs.ra.deg
            b = coord.icrs.dec.deg
        x = self.wcs.wcs_world2pix(l,b,1)
        return x[1],x[0]

    def angular_distance(self,i1,j1,i2,j2):
        # angular distance between pixel coordinates (I1,J1) and (I2,J2)
        c1 = self.pixel_coordinate(i1,j1)
        c2 = self.pixel_coordinate(i2,j2)
        d  = hp.rotator.angdist(coord2vec(c1),coord2vec(c2))*u.rad
        return d


# ------- POSTSTAMPS -------------

    def stamp_central_region(self,lado):
        c      = self.size[0]//2
        l      = lado//2
        d      = self.datos
        subcut = d[c-l:c+l,c-l:c+l]
        r      = Imagen(subcut,self.centro,np.array(subcut.shape),self.pixsize)
        r.image_coordsys = self.image_coordsys
        r.image_header = None
        return r

    def stamp_coord(self,coord,lado):

        imagen      = fits.PrimaryHDU(self.datos)
        wcs0        = wcs.WCS(self.header)
        wcs1        = wcs0.copy()
        cutout      = Cutout2D(imagen.data,
                               position=coord,
                               size=lado,
                               wcs=wcs1,
                               mode='partial')
        imagen.data = cutout.data
        imagen.header.update(cutout.wcs.to_header())
        output_img  = Imagen.from_hdu(imagen)

        return output_img


# ------- FILTERING -------------------------

    def matched(self,fwhm=1.0*u.deg,toplot=False):
        if np.size(fwhm) == 1:
            s = (fwhm/self.pixsize).si.value
            fdatos = mf.matched_filter(self.datos,lafwhm=s,
                                       nbins=self.lsize//4,
                                       aplot=False)
        else:
            fdatos = mf.matched_filter(self.datos,
                                       gprof=False,
                                       lafwhm=1.0,
                                       tprof0=img_shapefit(fwhm,self.datos),
                                       nbins=self.lsize//4,
                                       aplot=False)
        fmapa  = Imagen(fdatos,self.centro,self.size,self.pixsize)
        if toplot:
            fmapa.draw(newfig=True)
        return fmapa

    def iter_matched(self,fwhm=1.0*u.deg,toplot=False):
        s = (fwhm/self.pixsize).si.value
        fdat1,fdatos = mf.iterative_matched_filter(self.datos,
                                                   lafwhm=s,
                                                   nbins=self.lsize//4,
                                                   snrcut=5.0,
                                                   aplot=False,
                                                   topad=True,
                                                   kind='linear')
        fmapa  = Imagen(fdat1,self.centro,self.size,self.pixsize)
        fmapai = Imagen(fdatos,self.centro,self.size,self.pixsize)
        if toplot:
            fmapa.draw(newfig=True)
        return fmapa,fmapai

    def smooth(self,fwhm=1.0*u.deg,toplot=False):
        
        sigma  = (fwhm2sigma*fwhm/self.pixsize).si.value
        print(' --- Smoothing image with a Gaussian kernel of sigma = {0} pixels'.format(sigma))
        fdatos = gaussian_filter(self.datos,sigma=sigma)
        fmapa  = Imagen(fdatos,self.centro,self.size,self.pixsize)
        if toplot:
            fmapa.draw(newfig=True)
        return fmapa

# ------- FITTING -----------------------

    def central_gaussfit(self,return_output=False):
        patch = self.datos.copy()
        cfit  = fit_single_peak(patch)
        sigma = self.pixsize * cfit.sigma
        area  = (2*np.pi*sigma*sigma).si
        fwhm  = sigma2fwhm*sigma
        print(' --- Fitted beam area = {0}'.format(area))
        print(' --- Fitted beam fwhm = {0}'.format(fwhm))
        print(' --- Fitted amplitude = {0}'.format(cfit.amplitude))
        print(' --- Fitted centre    = ({0},{1})'.format(cfit.x,cfit.y))
        if return_output:
            return cfit


# ------- MASKING -----------------------

    def mask_value(self,value):
        d = self.datos
        self.datos = np.ma.masked_array(data=d,mask=d==value)

    def mask_brighter(self,value):
        d = self.datos
        self.datos = np.ma.masked_array(data=d,mask=d>value)

    def mask_fainter(self,value):
        d = self.datos
        self.datos = np.ma.masked_array(data=d,mask=d<value)

    def mask_border(self,nbpix):
        d = self.datos
        z = np.zeros(d.shape,dtype=bool)
        m = d.shape[0]
        z[0:nbpix,:]   = True
        z[m-nbpix:m,:] = True
        m = d.shape[1]
        z[:,0:nbpix]   = True
        z[:,m-nbpix:m] = True
        self.datos = np.ma.masked_array(data=d,mask=z)

    def mask_brightest_fraction(self,fraction):
        d = self.datos
        if np.ma.is_masked(d):
            x = d.flatten()
            y = x[x.mask==False]
            z = np.sort(y)
            s = round(fraction*z.size)
            v = z[-s]
        else:
            x = np.sort(d.flatten())
            s = round(fraction*x.size)
            v = x[-s]
        self.datos = np.ma.masked_array(data=d,mask=d>=v)

    def fraction_masked(self):
        return np.count_nonzero(self.datos.mask)/self.datos.size

# ------- PSF ----------------------------

    def psfmap(self,fwhm):
        fwhm_pix = (fwhm/self.pixsize).si.value
        g        = makeGaussian(self.size[0],fwhm=fwhm_pix)
        return Imagen(g,self.centro,self.size,self.pixsize)


# ------- PROJECTIONS -------------------

    @property
    def gnomic_projector(self):
        c = self.center_coordinate
        b = c.b.deg
        l = c.l.deg
        p = hp.projector.GnomonicProj(rot=[b,90.0-l],
                                      coord='G',xsize=self.size[0],
                                      ysize=self.size[1],
                                      reso=60*self.pixsize_deg)
        return p

# ------- RESAMPLING -------------------

    def downsample(self,factor=2,func=np.sum):
        data0 = self.datos
        data1 = block_reduce(data0,block_size=factor,func=func)
        return Imagen(data1,self.centro,
                      tuple(ti//factor for ti in self.size),
                      self.pixsize*factor)

    def upsample(self,factor=2,conserve_sum=True):
        data0 = self.datos
        data1 = block_replicate(data0,block_size=factor,conserve_sum=conserve_sum)
        return Imagen(data1,self.centro,
                      tuple(ti*factor for ti in self.size),
                      self.pixsize/factor)


# ------- EXAMPLE FILES -------------

example1 = '/Users/herranz/Trabajo/Test_Data/f001a066.fits'
