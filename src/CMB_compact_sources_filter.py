"""
    This program takes as input images of the CMB intensity in a fits file, the frequency
    and fwhm at which the map was observed and the ratio (cut) needed between
    a signal value and the background noise to consider the signal as compact
    source and filters them to locate compact sources mixed in the CMB. A
    catalogue of this sources is then created with important info of each source,
    its flux, coordinates in the sky and SNR. Finally .fits file with the catalogue
    of compact sources is given as an output.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import search_around_sky
import healpy as hp
import astropy.units as u
from fits_maps import Fitsmap
from astropy.coordinates import SkyCoord
from tqdm import trange
from skimage.feature import peak_local_max
from astropy.table import Table
import time
import warnings

warnings.filterwarnings("ignore")

def filter_cmb(path="/home/chema/Escritorio/datos/LFI_SkyMap_030-BPassCorrected_1024_R3.00_full.fits",
               f=30, fwhm=32.65, cut=4, print_sources=False):
    star_time = time.time()

    # Loading of the FITS file with the image of the CMB and change the units to Jy. Specify the path to the file,
    # the frequency and the fwhm of the instrument.
    f, fwhm = f * u.GHz, fwhm * u.arcmin  # Input data

    # The first element of the array is the one with the T map.
    object_map = Fitsmap.from_file(path, freq=f, fwhm=fwhm)[0]

    print("--- Data loaded ---")

    # Changing data from K to Jy
    object_map.to_Jy()

    # --- Parameters that characterize the patches that will be used to cover the CMB map on
    # which we will use the filter ---
    patch_size = 256

    # --- Computing the center of the patches to cover all the sphere ---
    nside = 8
    npix = hp.nside2npix(nside)

    centers = [hp.pix2vec(nside, i) for i in range(npix)]
    # List with the position vectors of the npix of side = 8 that will be the center of the patches.

    lon, lat = hp.vec2ang(np.array(centers), lonlat=True)  # Center vector to lon and lat in deg.

    npatches = len(centers)

    # --- Filtering every patch and getting all the candidates of compact sources, gathering all in a table ---
    peaks = []

    for i in trange(npatches):
        coord = SkyCoord(l=lon[i], b=lat[i], frame='galactic', unit=u.deg)
        patch = object_map.patch(coord, npix=patch_size)
        filtered_patch = patch.matched(fwhm=object_map.fwhm)
        std = filtered_patch.std()
        threshold = cut * std
        indices = peak_local_max(filtered_patch.datos, min_distance=2,
                                 threshold_abs=threshold, exclude_border=True, indices=True)
        for x, y in indices:
            coordinate = filtered_patch.pixel_coordinate(x, y)
            # Estimation of the flux error
            s = (object_map.fwhm/filtered_patch.pixsize).si.value
            w = np.arange(0, patch_size, 1, float)
            v = w[:, np.newaxis]
            r = np.sqrt((w - x) ** 2 + (v - y) ** 2)
            m = (r >= 3*s) & (r <= 5*s)

            peak = {'Longitude': coordinate.l.value,
                    'Latitude': coordinate.b.value,
                    'Flux': filtered_patch.datos[x, y],
                    'Flux_error': filtered_patch.datos[m].std(),
                    'SNR': filtered_patch.datos[x, y] / std}
            peaks.append(peak)

    print("--- Map filtered ---")

    table = Table(peaks)

    coordinates = SkyCoord(l=np.array(table['Longitude'].tolist()),
                           b=np.array(table['Latitude'].tolist()),
                           frame='galactic',
                           unit=u.deg)  # Coordinates of each peak in the galactic system (in angles).

    # --- Searching for repetitions in our catalogue ---

    resolution_max = object_map.fwhm.to(u.arcmin)

    ind1, ind2, dist, d3d = search_around_sky(coordinates, coordinates, 3/4*resolution_max)

    correspondence_table = Table()
    correspondence_table['Index 1'] = ind1
    correspondence_table['Index 2'] = ind2
    correspondence_table['Separation'] = dist

    final_table = Table()

    # --- Resolution of coincidences ---
    while len(correspondence_table) > len(final_table):
        skipped = []
        final_table = Table()
        for i in trange(len(table)):
            if i == 0:
                final_table = Table(rows=table[i], names=('Longitude', 'Latitude', 'Flux', 'Flux_error', 'SNR'))
            if i in skipped:
                continue
            x = correspondence_table[correspondence_table['Index 1'] == i]
            if len(x) == 1:
                if i == 0:
                    continue
                else:
                    final_table.add_row([table['Longitude'][i], table['Latitude'][i],
                                         table['Flux'][i], table['Flux_error'][i],
                                         table['SNR'][i]])
            else:
                skipped = np.append(skipped, np.array(x['Index 2'].tolist()))
                coord = SkyCoord(l=table['Longitude'].tolist()[i], b=table['Latitude'].tolist()[i],
                                 frame='galactic', unit=u.deg)
                patch = object_map.patch(coord, npix=int(patch_size/2), deltatheta_deg=14.658/2)
                filtered_patch = patch.matched(fwhm=object_map.fwhm)
                std = filtered_patch.std()
                threshold = cut * std
                indices = peak_local_max(filtered_patch.datos, min_distance=2,
                                         threshold_abs=threshold, exclude_border=True, indices=True)

                for x, y in indices:
                    s = (object_map.fwhm / filtered_patch.pixsize).si.value
                    if (x >= patch_size/4 - 3/4*s) & (x <= patch_size/4 + 3/4*s) & \
                            (y >= patch_size/4 - 3/4*s) & (y <= patch_size/4 + 3/4*s):
                        coordinate = patch.pixel_coordinate(x, y)

                        # Estimation of the flux error
                        w = np.arange(0, int(patch_size/2), 1, float)
                        v = w[:, np.newaxis]
                        r = np.sqrt((w - x) ** 2 + (v - y) ** 2)
                        m = (r >= 3 * s) & (r <= 5 * s)

                        peak = {'Longitude': coordinate.l.value,
                                'Latitude': coordinate.b.value,
                                'Flux': filtered_patch.datos[x, y],
                                'Flux_error': filtered_patch.datos[m].std(),
                                'SNR': filtered_patch.datos[x, y] / std}
                        final_table.add_row([peak['Longitude'], peak['Latitude'], peak['Flux'],
                                             peak['Flux_error'], peak['SNR']])
                        if i == 0:
                            final_table.remove_row(0)

        coordinates = SkyCoord(l=np.array(final_table['Longitude'].tolist()),
                               b=np.array(final_table['Latitude'].tolist()),
                               frame='galactic',
                               unit=u.deg)
        ind1, ind2, dist, d3d = search_around_sky(coordinates, coordinates, 3/4*resolution_max)

        correspondence_table = Table()
        correspondence_table['Index 1'] = ind1
        correspondence_table['Index 2'] = ind2
        correspondence_table['Separation'] = dist
        table = final_table

    print("--- Duplicated sources removed ---")

    # Saving the table with the catalogue.

    gal_coords = SkyCoord(l=np.array(table['Longitude'].tolist()), b=np.array(table['Latitude'].tolist()),
                          frame='galactic', unit=u.deg)
    eq_coords = gal_coords.icrs

    table.add_column(col=eq_coords.ra, index=2, name='RA')
    table.add_column(col=eq_coords.dec, index=2, name='DEC')
    table['Longitude'].unit, table['Latitude'].unit = 'deg', 'deg'
    table['Flux'].unit, table['Flux_error'].unit = 'Jy', 'Jy'
    table['RA'].unit, table['DEC'].unit = 'deg', 'deg'

    table.write('catalogue_' + str(f.value) + 'GHz.fits', overwrite=True)

    print("--- Data saved ---")

    # Saving a picture with the localization of all the sources.
    if print_sources:
        hp.projscatter(table['Longitude'].data.data,
                       table['Latitude'].data.data,
                       lonlat=True, coord='G', color='r', marker='.')
        hp.graticule()
        hp.projscatter(table['Longitude'].data.data,
                       table['Latitude'].data.data,
                       lonlat=True, coord='G', color='r', marker='.')
        plt.savefig('Detected_Sources_' + str(f) + '_SNR'+str(cut)+'.png')

    print(str(len(table))+' compact sources detected.')
    print('Execution time: ' + str(time.strftime("%M:%S", time.gmtime(time.time() - star_time))) + ' (min:s)')




