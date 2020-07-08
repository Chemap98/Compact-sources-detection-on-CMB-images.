This code was created by José María Palencia Sainz to obtain the degree in physics by the Universidad de Cantabria (UC).
This work was supervised by Diego Herranz Muñoz. 
(The main program (CMB_compact_sources_filter.py) was created by José María while the rest of programs were implemented by Diego)

***     Brief summary of the code     ***

This code filters images of the Cosmic Microwaves Background (CMB) intensity (I Stokes parameter) in order to obtain a catalogue of compact sources. This sources, mainly extragalactic sources, contaminate the CMB images and should be removed from this images in oder to study the CMB itself. On the other hand this sources are also interesting to study.

The code creates a catalogue with the detected compact sources. The catalogues include for every source its galactic and ecuatorial coordinates, an estimation of its flux with an uncertainty and its Signal to Noise Ratio (SNR).

In order to work this code needs the following things as input data:

    The path in the computer to a fits file with the CMB image.     Example given: "../data/ LFI_SkyMap_030-BPassCorrected_1024_R3.00_full.fits"
   
    The frequency at wich the image was taken.                      Example given: 28.4  (Planck RIMO, effective frecuency for the 30 GHz of the LFI.)

    The Full Width at Half Maximum. (Resolution of the image)       Example given: 32.293 (Panck RIMO, effective FWHM fo the 30 GHz of the LFI.)

    The minimum SNR of a source after the filtering.                Example given: 3.5 (Not recommended to go below this value. Usual values go from 5 to 4.)

    Boolean value to make an image of the location of the sources   Example given: print_sources=False


The output are the catalogues in a fits file. The name includes the frequency of the image used.    Example given: "../examples/catalogue_28.4GHz.fits"

This code uses a matched filter to detect the compact sources, it assumes that the sources in the image take a gaussian profile or at least one similar enough.
The code prints in the python console, or the text editor console, the step that has been completed, a load bar with the status of the current step and the estimated time remaining for each step and the total time the code has been running.

Some of the libraries that this code imports doesn't work on windows, this means that the code can only be executed on Mac IOS and Linux. The code has been written in Python3.

All the needed python files are included in the src folder, the main file is named "CMB_compact_sources_filter.py". This file contains a callabe funtion 'filter_CMB' that, given the input data, generates the final output.

***     Example of use      ***
      
        >>>filter_cmb(path="../data/LFI_SkyMap_030-BPassCorrected_1024_R3.00_full.fits", f=28.4, fwhm=32.293, cut=3.5, print_sources=False)

The data folder with all the maps used in this example (ESA: Planck mission 2015) is too big for this repository. This data can be downloaded in http://pla.esac.esa.int/pla/#maps.

