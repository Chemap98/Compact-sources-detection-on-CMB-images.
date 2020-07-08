#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:03:16 2017

@author: herranz
"""

import numpy as np
from numpy import dtype
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord
import scipy.stats.mstats as mst
import time
from astropy.units import UnitConversionError
from astropy.visualization import hist
from scipy.stats import normaltest,norm
import matplotlib.pyplot as plt
import re
from sky_images import Imagen
from utils import coord2vec,fwhm2sigma,sigma2fwhm

#### Example files, used for testing:

testfile  = '/Users/herranz/Trabajo/Planck/Data/PLA/LFI_SkyMap_030-BPassCorrected_0256_R2.01_full.fits'
testfile2 = '/Users/herranz/Dropbox/test.fits'
testfile3 = '/Users/herranz/Dropbox/testmask.fits'
Planck_masks_file = '/Users/herranz/Trabajo/Planck/Non_Thermal_Catalogue/Data/Planck_masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits'

class Fitsmap:

#    Class for Healpix maps and masks. Each FITMAPS instance
#     contains n-dimensional maps (DATA) with its corresponding
#     n-dimensional mask, plus a HEADER dictionary and an ordering
#     (RING or NEST) parameter.
#
#     The class is intended to facilitate Healpy programming,
#     in particular allowing for direct map arithmetic operations such
#     as  MAP3 = MAP1 * MAP2 taking into account masks
#     and masked map statistics
#
#     Important notice: the mask convention used in this class is
#     as follows:
#        mask == True   means the pixel has to be masked (equivalent to IDL mask = 0)
#        mask == False  means the pixel is not masked (equivalent to IDL mask=1)


    comment_count  = 0
    obs_frequency  = None
    fwhm_base      = None
    beam_area_base = None

    def __init__(self,data,mask,header,ordering):       # Instantiation
        self.data     = data
        self.mask     = mask
        self.header   = header
        self.ordering = ordering

    def __getitem__(self,sliced):                        # Slicing operation

        n = self.nmaps
        if n == 1:
            datos = self.data.copy()
            mask  = self.mask.copy()
        else:
            if isinstance(sliced,list) and len(sliced)==1:
                datos = self.data[sliced[0],:].copy()
                mask  = self.mask[sliced[0],:].copy()
            elif isinstance(sliced,str):
                elemento = None
                for nombre,valor in self.header.items():
                    if valor == sliced:
                        elemento = nombre
                if elemento is None:
                    print(' --- Warning: field not found in Fitsmap header')
                    datos = self.data.copy()
                    mask  = self.mask.copy()
                else:
                    islice = int(re.findall('\d+',elemento)[0])-1
                    datos  = self.data[islice,:].copy()
                    mask   = self.mask[islice,:].copy()
            else:
                datos = self.data[sliced,:].copy()
                mask  = self.mask[sliced,:].copy()

        ordering = self.check_order()
        header   = self.header.copy()

        if isinstance(sliced,list):
            nombre = [self.columns[slc] for slc in sliced]
            unidad = [self.units[slc] for slc in sliced]
        elif isinstance(sliced,str):
            nombre = self.columns[islice]
            unidad = self.units[islice]
        else:
            nombre = self.columns[sliced]
            unidad = self.units[sliced]

        for i in range(n):
            try:
                del header['TTYPE{0}'.format(i+1)]
            except KeyError:
                pass
            try:
                del header['TUNIT{0}'.format(i+1)]
            except KeyError:
                pass
            try:
                del header['TFORM{0}'.format(i+1)]
            except KeyError:
                pass

        if isinstance(nombre,list):
            for i in range(len(nombre)):
                header['TTYPE{0}'.format(i+1)] = nombre[i]
                header['TUNIT{0}'.format(i+1)] = unidad[i]
            header['TFIELDS'] = len(nombre)
        else:
            header['TTYPE1']  = nombre
            header['TUNIT1']  = unidad
            header['TFIELDS'] = 1

        outmap = Fitsmap(datos,mask,header,ordering)
        outmap.obs_frequency = self.obs_frequency
        if np.isscalar(self.fwhm):
            outmap.fwhm_base          = self.fwhm
            outmap.beam_area_base     = self.beam_area
        else:
            try:
                outmap.fwhm_base      = self.fwhm[sliced]
                outmap.beam_area_base = self.beam_area[sliced]
            except TypeError:
                outmap.fwhm_base      = self.fwhm
                outmap.beam_area_base = self.beam_area

        return outmap

### ---------   ARITHMETIC AND LOGICAL OPERATORS ---------------------------

    def __neg__(self):                                  # MINUS operator
        if self.ismask:
            r = Fitsmap(np.invert(self.data.copy()),
                        np.invert(self.mask.copy()),
                        self.header.copy(),self.check_order())
        else:
            r = Fitsmap(-1*self.data.copy(),
                        self.mask.copy(),
                        self.header.copy(),self.check_order())
        return r

    def __add__(self,other):                           # SUM operator

        compatible,ctype = self.check_compatibility(other)
        header = self.header.copy()
        if compatible:
            if ctype == 'map':
                self.to_ring()
                other.to_ring()
                ordering = 'RING'
                datos  = self.data.copy() + other.data.copy()
                mask   = self.mask.copy() + other.mask.copy()
            elif ctype == 'scalar':
                datos    = self.data.copy() + other
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == '1Darray':
                datos    = self.data.copy() + other[0]
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'vector':
                datos = self.data.copy()
                for j in range(self.nmaps):
                    datos[j,:] += other[j]
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'mask':
                self.to_ring()
                other.to_ring()
                ordering = 'RING'
                datos = self.data.copy()
                mask  = self.mask.copy() + other.mask.copy()
            else:
                datos    = self.data.copy()
                mask     = self.mask.copy()
                ordering = self.check_order()
        else:
            datos    = self.data.copy()
            mask     = self.mask.copy()
            ordering = self.check_order()

        return Fitsmap(datos,mask,header,ordering)

    def __radd__(self, other):                         # reverse SUM
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self,other):                           # SUBTRACT operator
        compatible,ctype = self.check_compatibility(other)
        header = self.header.copy()
        if compatible:
            if ctype == 'map':
                self.to_ring()
                other.to_ring()
                ordering = 'RING'
                datos  = self.data.copy() - other.data.copy()
                mask   = self.mask.copy() + other.mask.copy()
            elif ctype == 'scalar':
                datos    = self.data.copy() - other
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == '1Darray':
                datos    = self.data.copy() - other[0]
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'vector':
                datos = self.data.copy()
                for j in range(self.nmaps):
                    datos[j,:] -= other[j]
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'mask':
                self.to_ring()
                other.to_ring()
                ordering = 'RING'
                datos = self.data.copy()
                mask  = self.mask.copy() + other.mask.copy()
            else:
                datos    = self.data.copy()
                mask     = self.mask.copy()
                ordering = self.check_order()
        else:
            datos    = self.data.copy()
            mask     = self.mask.copy()
            ordering = self.check_order()

        return Fitsmap(datos,mask,header,ordering)

    def __rsub__(self, other):                         # reverse SUBTRACT
        if other == 0:
            return self
        else:
            return self.__sub__(other)

    def __mul__(self,other):                           # MULTIPLY operator
        compatible,ctype = self.check_compatibility(other)
        header = self.header.copy()
        if compatible:
            if ctype == 'map':
                self.to_ring()
                other.to_ring()
                ordering = 'RING'
                datos  = np.multiply(self.data.copy(),other.data.copy())
                mask   = self.mask.copy() * other.mask.copy()
            elif ctype == 'scalar':
                datos    = self.data.copy() * other
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == '1Darray':
                datos    = self.data.copy() * other
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'vector':
                datos = self.data.copy()
                for j in range(self.nmaps):
                    datos[j,:] = other[j]*datos[j,:]
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'mask':
                self.to_ring()
                other.to_ring()
                ordering = 'RING'
                mask  = self.mask.copy() * other.mask.copy()
                if self.ismask and not other.ismask:
                    datos = other.data.copy()
                elif self.ismask and other.ismask:
                    datos = mask
                else:
                    datos = self.data.copy()
            else:
                datos    = self.data.copy()
                mask     = self.mask.copy()
                ordering = self.check_order()
        else:
            datos    = self.data.copy()
            mask     = self.mask.copy()
            ordering = self.check_order()

        return Fitsmap(datos,mask,header,ordering)

    def __rmul__(self, other):                        # reverse MULTIPLY
        if other == 0:
            return self
        else:
            return self.__mul__(other)

    def __pow__(self,p):                              # POWER operator
        compatible,ctype = self.check_compatibility(p)
        header = self.header.copy()
        if compatible:
            if ctype == 'map':
                self.to_ring()
                p.to_ring()
                ordering = 'RING'
                datos  = np.power(self.data.copy(),p.data.copy())
                mask   = self.mask.copy() + p.mask.copy()
            elif ctype == 'scalar':
                datos    = np.power(self.data.copy(),p)
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == '1Darray':
                datos    = np.power(self.data.copy(),p[0])
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'vector':
                datos = self.data.copy()
                for j in range(self.nmaps):
                    datos[j,:] = np.power(datos[j,:],p[j])
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'mask':
                self.to_ring()
                p.to_ring()
                ordering = 'RING'
                mask  = self.mask.copy() + p.mask.copy()
                if self.ismask:
                    datos = mask
                else:
                    datos = self.data.copy()
            else:
                datos    = self.data.copy()
                mask     = self.mask.copy()
                ordering = self.check_order()
        else:
            datos    = self.data.copy()
            mask     = self.mask.copy()
            ordering = self.check_order()

        return Fitsmap(datos,mask,header,ordering)

    def check_compatibility(self,other):              # checks if SELF is
                                                      # compatible
                                                      # with OTHER for
                                                      # arithmetic operations

        compatibility = False
        ctype         = None

        comp_type = type(other) == type(self)
        if comp_type:
            if self.data.shape == other.data.shape:
                compatibility = True
                if self.ismask or other.ismask:
                    ctype = 'mask'
                else:
                    ctype = 'map'
            else:
                compatibility = False
                print(' --- Warning: fits data dimensions do not match')
        else:
            if (type(other)==int) or (type(other)==float) or (type(other)==complex):
                compatibility = True
                ctype = 'scalar'
            if np.size(other)==1 and ctype!='scalar':
                try:
                    dt = other.dtype
                    if dt == dtype('float64') or dt == dtype('int64') or dt == dtype('complex128'):
                        compatibility = True
                        ctype = '1Darray'
                except AttributeError:
                    pass
            if isinstance(other,np.ndarray) and np.size(other)>1 and np.size(other)==self.nmaps:
                compatibility = True
                ctype = 'vector'

        if not compatibility:
            print(' --- Warning: unable to find object compatibility')
            ctype = 'none'

        return compatibility,ctype


### ---------   COPY METHOD -------------------------------------------------

    def check_order(self):
        if self.ordering.upper() == 'RING':
            order = 'RING'
        else:
            order = 'NEST'
        return order

    def copy(self):
        copia = Fitsmap(self.data.copy(),self.mask.copy(),
                        self.header.copy(),self.check_order())

        copia.comment_count  = self.comment_count
        copia.obs_frequency  = self.obs_frequency
        copia.fwhm_base      = self.fwhm
        copia.beam_area_base = self.beam_area
        return copia

### ---------   MASKS -------------------------------------------------------

    def mask_value(self,value):
        self.mask = self.data == value

    def masked_data(self,i=-1):
        if (self.nmaps == 1) or (i<0):
            v = np.ma.array(self.data.copy(),mask=self.mask.copy())
        else:
            v = np.ma.array(self.data[i,:].copy(),mask=self.mask[i,:].copy())
        return v

    def add_mask(self,mask):
        self.mask = self.mask + mask

    def mask_band(self,band_size): # masks a band of size band_size around
                                   # the Equator

        if isinstance(band_size,u.quantity.Quantity):
            t1 = band_size
        else: # assume the value is in degrees
            t1 = band_size*u.deg

        tuno = np.pi/2-t1.to(u.rad).value
        tdos = np.pi/2+t1.to(u.rad).value

        isnest = False
        if self.ordering.upper() == 'NEST':
            isnest = True

        ipix = hp.query_strip(self.nside,tuno,tdos,nest=isnest)
        n    = self.nmaps

        if n==1:
            self.mask[ipix] = True
            if self.ismask:
                self.data[ipix] = True
        else:
            for i in range(n):
                self.mask[i,ipix] = True
                if self.ismask:
                    self.data[i,ipix] = True

    def grow_mask_1pix(self):   # grows the mask by one pixel
        if self.nmaps == 1:
            mask   = self.mask.copy()
        else:
            mask   = self.mask[0,:].copy()
        if self.ordering == 'NEST':
            nested = True
        else:
            nested = False
        listpix = [i for i in range(self.npix) if not mask[i]]
        for pixel in listpix:
            ipix = hp.get_all_neighbours(self.nside,pixel,nest=nested)
            n    = np.count_nonzero(mask[ipix])
            if n>0:
                if self.nmaps == 1:
                    self.mask[pixel] = True
                else:
                    self.mask[:,pixel] = True

    def grow_mask_radius(self,radius):   # grows the mask by a certain radius
        if self.nmaps == 1:
            mask   = self.mask.copy()
        else:
            mask   = self.mask[0,:].copy()
        if self.ordering == 'NEST':
            nested = True
        else:
            nested = False
        listpix = [i for i in range(self.npix) if not mask[i]]
        for pixel in listpix:
            v    = hp.pix2vec(self.nside,pixel,nest=nested)
            ipix = hp.query_disc(self.nside,v,radius.to(u.rad).value,
                                 inclusive=True,
                                 nest=nested)
            n    = np.count_nonzero(mask[ipix])
            if n>0:
                if self.nmaps == 1:
                    self.mask[pixel] = True
                else:
                    self.mask[:,pixel] = True

    def to_mask(self):
        self.data = self.mask.copy()

    @property
    def mask_area(self):
        return np.count_nonzero(self.mask)*self.pixel_area

    @property
    def unmasked_area(self):
        return 4.0*np.pi*u.sr-self.mask_area

    @property
    def masked_fraction(self):
        return np.count_nonzero(self.mask)/self.npix

    @property
    def unmasked_fraction(self):
        return 1.0-self.masked_fraction

### ---------   SMOOTHING ---------------------------------------------------

    def smooth(self,fwhm):
        mapa = self.copy()
        mapa.to_ring()
        if mapa.nmaps == 1:
            x = mapa.data
            x = hp.smoothing(x,fwhm=fwhm.to(u.rad).value,pol=False)
            mapa.data = x
            mapa.fwhm_base = np.sqrt((mapa.fwhm_base**2+fwhm**2).si)
        elif mapa.nmaps == 3:
            x = [mapa.data[0,:],mapa.data[1,:],mapa.data[2,:]]
            x = hp.smoothing(x,fwhm=fwhm.to(u.rad).value,pol=True)
            mapa.data = np.array(x)
            mapa.fwhm_base = np.sqrt((mapa.fwhm_base**2+fwhm**2).si)
        else:
            for i in range(mapa.nmaps):
                x = mapa.data[i,:]
                x = hp.smoothing(x,fwhm=fwhm.to(u.rad).value,pol=False)
                mapa.data[i,:] = x
                mapa[i,:].fwhm_base = np.sqrt((mapa[i,:].fwhm_base**2+fwhm**2).si)
        mapa.ordering = 'RING'
        return mapa

### ---------   STATISTICS ---------------------------------------------------

    def ngood(self,i=-1):
        if i>=0:
            a  = self.masked_data(i)
            x  = a.data
            m1 = np.invert(a.mask)
            m2 = hp.mask_good(x,self.badval)
            r  = np.count_nonzero(m1*m2)
        else:
            r = []
            for k in range(self.nmaps):
                a  = self.masked_data(k)
                x  = a.data
                m1 = np.invert(a.mask)
                m2 = hp.mask_good(x,self.badval)
                r.append(np.count_nonzero(m1*m2))
            r = np.array(r)
        return r

    def maxval(self,i=-1):
        if i>=0:
            a = self.masked_data(i)
            r = a.max()
        else:
            r = []
            for k in range(self.nmaps):
                a = self.masked_data(k)
                r.append(a.max())
            r = np.array(r)
        return r

    def minval(self,i=-1):
        if i>=0:
            a = self.masked_data(i)
            r = a.min()
        else:
            r = []
            for k in range(self.nmaps):
                a = self.masked_data(k)
                r.append(a.min())
            r = np.array(r)
        return r

    def mean(self,i=-1):
        if i>=0:
            a = self.masked_data(i)
            r = a.mean()
        else:
            r = []
            for k in range(self.nmaps):
                a = self.masked_data(k)
                r.append(a.mean())
            r = np.array(r)
        return r

    def std(self,i=-1):
        if i>=0:
            a = self.masked_data(i)
            r = a.std()
        else:
            r = []
            for k in range(self.nmaps):
                a = self.masked_data(k)
                r.append(a.std())
            r = np.array(r)
        return r

    def skew(self,i=-1,**kwargs):
        if i>=0:
            a = self.masked_data(i)
            r = np.asscalar(mst.skew(a,**kwargs).data)
        else:
            r = []
            for k in range(self.nmaps):
                a = self.masked_data(k)
                r.append(np.asscalar(mst.skew(a,**kwargs).data))
            r = np.array(r)
        return r

    def kurtosis(self,i=-1,**kwargs):

        if i>=0:
            a = self.masked_data(i)
            r = mst.kurtosis(a,**kwargs)
        else:
            r = []
            for k in range(self.nmaps):
                a = self.masked_data(k)
                r.append(mst.kurtosis(a,**kwargs))
            r = np.array(r)
        return r

    def statistics(self,i=-1):
        return {'mean':self.mean(i=i),
                'std':self.std(i=i),
                'max':self.maxval(i=i),
                'min':self.minval(i=i),
                'skew':self.skew(i=i),
                'kurt':self.kurtosis(i=i),
                'ngood':self.ngood(i=i)}

    def test_normal(self,i=-1,toplot=False,tofile=None,threshold=1.e-3):
        if self.nmaps==1:
            x = self.data.copy()
            m = self.mask.copy()
            y = x[np.invert(m)]
            statistic,pvalue = normaltest(y,nan_policy='omit')
            if pvalue<threshold:
                isnormal = False
            else:
                isnormal = True
        elif isinstance(i,int):
            x = self.data[i,:].copy()
            m = self.mask[i,:].copy()
            y = x[np.invert(m)]
            statistic,pvalue = normaltest(y,nan_policy='omit')
            if pvalue<threshold:
                isnormal = False
            else:
                isnormal = True
        elif isinstance(i,np.ndarray):
            toplot = False
            tofile = None
            statistic = []
            pvalue    = []
            isnormal  = []
            for j in range(self.nmaps):
                x = self.data[j,:].copy()
                m = self.mask[j,:].copy()
                y = x[np.invert(m)]
                stat,p = normaltest(y,nan_policy='omit')
                statistic.append(stat)
                pvalue.append(p)
                if p<threshold:
                    isnormal.append(False)
                else:
                    isnormal.append(True)

        if tofile is not None:
            toplot = True

        if toplot:
            columns = self.columns
            plt.figure()
            h = hist(y,bins='knuth',
                     histtype='stepfilled',
                     alpha=0.2,
                     normed=True,
                     color='blue',
                     label=columns[i])
            lax = h[1]
            m   = y.mean()
            s   = y.std()
            lay = norm.pdf(lax,m,s)
            plt.plot(lax,lay,'r')
            plt.xlabel(columns[i])
            plt.ylabel('normalized histogram')
            if tofile is not None:
                plt.savefig(tofile)


        return {'statistics':statistic,'p-value':pvalue,'is_normal':isnormal}


### ---------   PIXEL OPERATIONS AND COORDINATES ----------------------------

    def to_ring(self):
        if self.ordering.upper() == 'NEST':
            self.ordering = 'RING'
            self.data     = hp.reorder(self.data,n2r=True)
            self.mask     = hp.reorder(self.mask,n2r=True)
            self.header['ORDERING'] = 'RING'

    def to_nest(self):
        if self.ordering.upper() == 'RING':
            self.ordering = 'NEST'
            self.data     = hp.reorder(self.data,r2n=True)
            self.mask     = hp.reorder(self.mask,r2n=True)
            self.header['ORDERING'] = 'NEST'

    def pixel_to_coordinates(self,i):
        cframe = self.coordsys
        if cframe == 'G':
            framsys = 'galactic'
        if cframe == 'C':
            framsys = 'icrs'
        if cframe == 'E':
            framsys = 'geocentrictrueecliptic'
        if self.ordering.upper() == 'NEST':
            innest = True
        else:
            innest = False

        lon,lat = hp.pix2ang(self.nside,i,nest=innest,lonlat=True)

        return SkyCoord(lon*u.deg,lat*u.deg,frame=framsys)

    def coordinates_to_pixel(self,coord):
        cframe = self.coordsys
        if cframe == 'C':
            lon = coord.icrs.ra.deg
            lat = coord.icrs.dec.deg
        elif cframe == 'E':
            lon = coord.geocentrictrueecliptic.lon.deg
            lat = coord.geocentrictrueecliptic.lat.deg
        else:
            lon = coord.galactic.l.deg
            lat = coord.galactic.b.deg
        nested = False
        if self.ordering.upper() == 'NEST':
            nested = True
        pix = hp.ang2pix(self.nside,lon,lat,nest=nested,lonlat=True)
        return pix

    def ud_grade(self,nside_new):
        nmap = hp.ud_grade(self.data.copy(),nside_out=nside_new,
                           order_in=self.check_order(),
                           order_out=self.check_order())
        nmask = hp.ud_grade(self.mask.copy(),nside_out=nside_new,
                            order_in=self.check_order(),
                            order_out=self.check_order())
        self.header['NSIDE'] = nside_new
        return Fitsmap(nmap,nmask,self.header.copy(),self.check_order())

    def disc_around_coordinates(self,coord,radius):
        pix = self.coordinates_to_pixel(coord)
        nested = False
        if self.ordering.upper() == 'NEST':
            nested = True
        vec = hp.pix2vec(self.nside,pix,nest=nested)
        return hp.query_disc(self.nside,vec,radius.to(u.rad).value,
                             inclusive=True,nest=nested)

    def fraction_masked_disk(self,coord,radius):
        n_masked = 0
        n_total  = 0
        for i in range(self.nmaps):
            temp     = self[i]
            inx      = temp.disc_around_coordinates(coord,radius)
            n_masked += np.count_nonzero(temp.mask[inx])
            n_total  += temp.data[inx].size
        return float(n_masked)/float(n_total)

    @property
    def pixel_area(self):
        return hp.nside2pixarea(self.nside)*u.sr

### ---------   VISUALIZATION  -----------------------------------------------

    def moll(self,i=0,tofile=None,**kwargs):

        c = self.columns
        n = self.nmaps
        if n > 1:
            x = self.data[i,:].copy()
            m = self.mask[i,:].copy()
        else:
            x = self.data.copy()
            m = self.mask.copy()

        if np.count_nonzero(m)>0:
            x[m] = hp.UNSEEN

        hp.mollview(x,title=c[i]+' ['+self.units[i]+']',**kwargs)

        if tofile is not None:
            plt.savefig(tofile)


### ---------   PATCHING  ----------------------------------------------------

    def patch(self,coord,npix=None,
              psi_deg=0.0,deltatheta_deg=14.658,
              toplot=False,tofile=None):

        if self.ordering.upper() == 'NEST':
            innest = True
        else:
            innest = False

        pixel      = self.coordinates_to_pixel(coord)
        theta,phi  = hp.pix2ang(self.nside,pixel,nest=innest)
        theta      = np.rad2deg(theta)
        phi        = np.rad2deg(phi)
        if npix is None:
            xsize = self.nside//4
        else:
            xsize = npix

        if self.fraction_masked_disk(coord,deltatheta_deg*u.deg/np.sqrt(2))>=0.9:

            img  = Imagen(np.zeros((npix,npix)),
                          np.array([theta,phi]),
                          np.array([npix,npix]),
                          deltatheta_deg/npix*u.deg)
        else:

#            psiz_deg = deltatheta_deg/(self.nside/4)
#            psiz_min = 60.0*psiz_deg
#            objeto   = hp.gnomview(self.data,rot=(phi,90.0-theta,psi_deg),
#                                   reso=psiz_min, xsize=self.nside//4,
#                                   flip="astro",coord=self.coordsys,
#                                   return_projected_map=True)

            psiz_deg = deltatheta_deg/(xsize)
            psiz_min = 60.0*psiz_deg
            objeto   = hp.gnomview(self.data,
#                                   no_plot=True,
                                   rot=(phi,90.0-theta,psi_deg),
                                   reso=psiz_min, xsize=xsize,
                                   flip="astro",coord=self.coordsys,
                                   return_projected_map=True)

            plt.close()
            datos    = objeto.data
            datos    = np.flipud(datos)
            s = datos.shape
            N1 = s[0]
            N2 = s[1]

            img  = Imagen(datos,
                          np.array([theta,phi]),
                          np.array([N1,N2]),
                          deltatheta_deg/N1*u.deg)

        if npix < img.lsize:
            imag = img.stamp_central_region(npix)
        else:
            imag = img

        if toplot:
            imag.draw(newfig=True)

        return imag

### ---------   BEAM AT GIVEN COORDINATES  -----------------------------------

    def spherical_beam(self,coordinate,bls,thetamax=10*u.deg):

        if self.ordering.upper() == 'RING':
            nest = False
        else:
            nest = True

        v1       = coord2vec(coordinate)
        nside2   = 4*self.nside
        npix2    = hp.nside2npix(nside2)
        beam_map = np.zeros(npix2)

        listpix = hp.query_disc(nside2,v1,
                                thetamax.to(u.rad).value,
                                inclusive=True,
                                nest=nest)

        for pixel in listpix:
            v2 = hp.pix2vec(nside2,pixel,nest=nest)
            d  = hp.rotator.angdist(v1,v2)*u.rad
            d  = d.value
            beam_map[pixel] = hp.bl2beam(bls,d)

        beam_map = hp.ud_grade(beam_map,self.nside,order_in=self.ordering)
        beam_map = beam_map/beam_map.max()

        return Fitsmap(beam_map,self.mask,self.header,self.ordering)

    def flat_beam(self,coordinate,bls,npix=512,deltatheta_deg=14.658):

        p0   = self.patch(coordinate,
                          npix*4,
                          deltatheta_deg=deltatheta_deg)

        dmap = np.zeros(p0.datos.shape)
        cent = [x//2 for x in p0.size]
        for i in range(npix*4):
            for j in range(i,npix*4):
                d         = p0.angular_distance(cent[0],cent[1],i,j)[0].to(u.rad).value
                dmap[i,j] = d
                dmap[j,i] = d
        p1   = Imagen(hp.bl2beam(bls,dmap),p0.centro,p0.size,p0.pixsize)
        p2   = p1.downsample(factor=4,func=np.mean)

        return p2


### ---------   UNIT CONVERSIONS ---------------------------------------------


    def to_unit(self,final_unit,barea=None,freq=None):

        input_units = self.physical_units

        if freq is None:
            freq      = self.obs_frequency

        if barea is None:
            beam_area = self.beam_area
        else:
            beam_area = barea

        if self.nmaps == 1:
            try:
                x = 1.0*input_units[0]
                t = x.to(final_unit,
                         equivalencies=u.brightness_temperature(beam_area,
                                                                freq))
                self.data[:] = t*self.data[:]
                self.header['TUNIT{0}'.format(1)] = final_unit.to_string()
            except UnitConversionError:
                try:
                    x = 1.0*input_units[0]**(1/2)
                    t = x.to(final_unit,
                             equivalencies=u.brightness_temperature(beam_area,
                                                                    freq))
                    self.data[:] = t*t*self.data[:]
                    self.header['TUNIT{0}'.format(1)] = final_unit.to_string()+'^2'
                except UnitConversionError:
                    pass
        else:
            for jmap in range(self.nmaps):
                try:
                    x = 1.0*input_units[jmap]
                    t = x.to(final_unit,
                             equivalencies=u.brightness_temperature(beam_area[jmap],
                                                                    freq))
                    self.data[jmap,:] = t*self.data[jmap,:]
                    self.header['TUNIT{0}'.format(jmap+1)] = final_unit.to_string()
                except UnitConversionError:
                    try:
                        x = 1.0*input_units[jmap]**(1/2)
                        t = x.to(final_unit,
                                 equivalencies=u.brightness_temperature(beam_area[jmap],
                                                                        freq))
                        self.data[jmap,:] = t*t*self.data[jmap,:]
                        self.header['TUNIT{0}'.format(jmap+1)] = '('+final_unit.to_string()+')^2'
                    except UnitConversionError:
                        pass

    def to_Jy(self,barea=None,freq=None):
        self.to_unit(u.Jy,barea=barea,freq=freq)

    def to_K(self,barea=None,freq=None):
        self.to_unit(u.K,barea=barea,freq=freq)

### ---------   HEADER MANIPULATION ------------------------------------------

    def set_name(self,column,nombre):
        self.header['TTYPE{0}'.format(column+1)] = nombre

    def set_unit(self,column,nombre):
        self.header['TUNIT{0}'.format(column+1)] = nombre

    def add_comment(self,comment):
        self.comment_count += 1
        self.header['COMMENT{0}'.format(self.comment_count)] = comment

    def locate_type(self,tstring):
        return [i for i in range(self.nmaps) if (self.header['TTYPE{0}'.format(i+1)] == tstring)]

### ---------   INPUT/OUTPUT -------------------------------------------------

    def write(self,filename,new_type=None,new_unit=None,additional_header=None):

        if new_type is not None:
            self.set_name(0,new_type)
        if new_unit is not None:
            self.set_unit(0,new_unit)

        columns  = self.columns
        if columns == ['']:
            columns =[]
            for i in range(self.nmaps):
                columns.append('col{0}'.format(i+1))
        units    = self.units
        coordsys = self.coordsys
        hextr    = []
        hextr.append(('COMMENT{0}'.format(self.comment_count),'--- Created by D. Herranz ----'))
        if self.ordering.upper() == 'NEST':
            nestb = True
        else:
            nestb = False

        try:
            pconv = self.header['POLCCONV']
            hextr.append(('POLCCONV',pconv))
        except KeyError:
            hextr.append(('POLCCONV','COSMO'))

        try:
            pconv = self.header['BAD_DATA']
            hextr.append(('BAD_DATA',pconv))
        except KeyError:
            hextr.append(('BAD_DATA',self.badval))

        hextr.append(('CREATION',time.asctime()))

        if additional_header is not None:
            hextr += additional_header

        if self.ismask:
            hp.write_map(filename,
                         self.data.astype('float'),
                         nest = nestb,
                         coord = coordsys,
                         extra_header = hextr,
                         overwrite = True)
        else:
            hp.write_map(filename,
                         np.ma.array(data=self.data,
                                     mask=self.mask,
                                     fill_value=self.badval),
                         nest = nestb,
                         coord = coordsys,
                         column_names = columns,
                         column_units = units,
                         extra_header = hextr,
                         overwrite = True)

    @classmethod
    def empty(self,n):
        x = np.zeros(hp.nside2npix(n))
        f = 'temp.fits'
        #f = '/Users/herranz/Dropbox/Temp/temp.fits'
        hp.write_map(f,x,overwrite=True)
        return read_healpix_map(f,verbose=False)

    @classmethod
    def from_file(self,fname,freq=None,fwhm=None):
        maps = read_healpix_map(fname,verbose=True,freq=freq,fwhm=fwhm)
        return maps

### ---------   OBJECT PROPERTIES ------------------------------------------

    @property
    def fwhm(self):
        if self.fwhm_base is None:
            if self.beam_area_base is None:
                output = None
            else:
                try:
                    nf = len(self.beam_area_base)
                except TypeError:
                    nf = 1
                if nf>1:
                    output = [sigma2fwhm*(np.sqrt(b/(2*np.pi))).si for b in self.beam_area_base]
                else:
                    output = self.nmaps*[sigma2fwhm*(np.sqrt(self.beam_area_base/(2*np.pi))).si]
                self.fwhm_base = output
        else:
            if self.nmaps > 1:
                try:
                    nf = len(self.fwhm_base)
                except TypeError:
                    nf = 1
                if nf == 1:
                    output = [self.fwhm_base]*self.nmaps
                elif nf != self.nmaps:
                    output = [self.fwhm_base[0]]*self.nmaps
                else:
                    output = self.fwhm_base
                output = np.array([g.si.value for g in output])*output[0].si.unit
            else:
                output = self.fwhm_base

            self.fwhm_base = output

        output = output

        return output

    @property
    def beam_area(self):
        if self.beam_area_base is None:
            if self.fwhm is None:
                a = self.pixel_area
                print(' Map beam area not found. Using pixel area instead ')
            else:
                f = self.fwhm
                s = fwhm2sigma*f
                a = 2.0*np.pi*(s*s).si
        else:
            a = self.beam_area_base
          
        try:
            if a.size != self.nmaps:
                a = self.nmaps*[a]
        except AttributeError:
            pass
            
        self.beam_area_base = a
        return a

    @property
    def T(self):  # Returns the first map (assumed to be T)
        if self.data.size == self.npix:
            return self.data
        else:
            return self.data[0,:]

    @property
    def Q(self): # Returns the second map (assumed to be Q), if it exists
        if self.data.size == self.npix:
            return np.array([])
        else:
            return self.data[1,:]

    @property
    def U(self): # Returns the third map (assumed to be U), if it exists
        if self.data.size == self.npix:
            return np.array([])
        else:
            return self.data[2,:]

    @property
    def badval(self): # Returns the bad pixel value
        try:
            b = self.header['BAD_DATA']
        except KeyError:
            self.header['BAD_DATA'] = hp.UNSEEN
            b = hp.UNSEEN
        return b

    @property
    def nmaps(self): # Returns number of maps in the Fitsmap
        if self.data.size == self.npix:
            n = 1
        else:
            n = self.data.shape[0]
        return n

    @property
    def nside(self): # Returns the NSIDE parameter
        return hp.get_nside(self.data)

    @property
    def npix(self):  # Returns the NPIXEL parameter
        return hp.pixelfunc.nside2npix(self.nside)

    @property
    def resolution(self):  # Returns the approximate pixel size, in arcmin
        return (hp.nside2resol(self.nside,arcmin=True))*u.arcmin

    @property
    def coordsys(self):  # Returns the coordinate system (by default, Galactic)
        try:
            csys = self.header['COORDSYS']
            if csys.upper() == 'EQUATORIAL':
                csys = 'C'
            else:
                csys = csys[0].upper()
        except KeyError:
            csys = 'G'
            self.header['COORDSYS'] = 'GALACTIC'
        return csys

    @property
    def columns(self):  # Returns, as a list of strings, the names of the maps
        n      = self.nmaps
        cnames = []
        for i in range(n):
            st = 'TTYPE{0}'.format(i+1)
            try:
                s = self.header[st]
            except KeyError:
                s = ''
                self.header[st] = ''
            cnames.append(s)
        return cnames

    @property
    def set_beam_areas(self):
        sigmas = fwhm2sigma*self.fwhm
        self.beam_area_base = 2.0*np.pi*(sigmas**2).si


    @property
    def units(self):  # Returns, as a list of strings, the units of the maps
        n      = self.nmaps
        cnames = []
        for i in range(n):
            st = 'TUNIT{0}'.format(i+1)
            try:
                s = self.header[st]
            except KeyError:
                s = ''
                self.header[st] = ''
            cnames.append(s)
        return cnames

    @property
    def physical_units(self): # Tries to return the units (as astropy.units
                              # quantities)
                              # of the maps. If the method in unable to find the
                              # unit, it returns an empty list in its place

        unit_string = self.units
        unit_phys   = []
        
        for unidad in unit_string:    
            if unidad == '' or unidad == ' ':
                unit_phys.append([])
            else:
                try:
                    x = unidad.replace('(K_CMB)','K')
                    x = x.replace('K_CMB','K')
                    g = u.Unit(x)
                    unit_phys.append(g)
                except ValueError:                
                    unit_phys.append([]) 
                    
        return unit_phys

    @property
    def ismask(self):   # Checks if the data in a Fitsmap is a boolean mask
        return self.data.dtype == np.dtype('bool')


"""
===============================================================================

"""

def read_healpix_map(fname,maskmap=False,freq=None,fwhm=None,maskval=None,verbose=True):

    if maskmap:
        maps,h = hp.read_map(fname,field=None,h=True,verbose=verbose)
        maps   = np.invert(maps.astype(np.bool))        # set mask = True where the pixel value is zero, False where the pixel value is one
    else:
        maps,h = hp.read_map(fname,field=None,h=True,verbose=verbose)

    ordering = 'RING'
    header   = dict(h)
    header['ORDERING'] = 'RING'

    if maskmap:
        mask      = maps
    elif maskval is not None:
        mask = maps==maskval
    else:
        try:
            bdata = header['BAD_DATA']
            mask  = maps==bdata
        except KeyError:
            mask  = np.zeros(maps.shape,dtype='bool')

    mapa = Fitsmap(maps,mask,header,ordering)

    mapa.obs_frequency = freq
    mapa.fwhm_base     = fwhm
    if fwhm is not None:
        sigma          = fwhm/(2*np.sqrt(2*np.log(2.0)))
        sigma          = sigma.to(u.rad)
        mapa.beam_area_base = 2*np.pi*sigma*sigma
    else:
        mapa.beam_area_base = None

    return mapa