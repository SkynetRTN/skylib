from skylib.astrometry import solve_field
from skylib.extraction import extract_sources
from astropy.io import fits
from numpy import arctan2, degrees, float32, transpose
import math

import logging

logging.basicConfig(level=logging.INFO)

# Example test case for the solve_field function
image_path = "scripts/example_image.fits"

hdul = fits.open(image_path)

# The first HDU is usually the Primary HDU
hdu = hdul[0]

# Get the image data as a NumPy ndarray
image_data = hdu.data
ra = hdu.header['CRVAL1']
dec = hdu.header['CRVAL2']
width = hdu.header['NAXIS1']
height = hdu.header['NAXIS2']
pixel_scale = math.sqrt(hdu.header['CD1_1']**2 + hdu.header['CD2_2']**2)  # degrees per pixel

sources = extract_sources(image_data, threshold=5, deblend=False, fwhm=0)[0]
n_field = len(sources)
if not n_field:
    logging.info('No sources detected')
else:
    logging.info('%d sources detected', n_field)
    if n_field < 4:
        logging.info('Too few sources for solving')
    else:
        xy = transpose([sources['x'], sources['y']])
        try:
            flux = sources['flux']
        except KeyError:
            try:
                flux = 10**(-sources['mag']/2.5)
            except KeyError:
                flux = None

        args = dict(flux=flux, width=width, height=height, ra_hours=ra/15.0, dec_degs=dec, radius=1, pixel_scale=pixel_scale, image_path=image_path)
        logging.info(
            'Solving field; %s',
            ', '.join('{}={}'.format(name, repr(val))
                        for name, val in args.items()))
        sol = solve_field(xy=xy, **args)
        if sol.wcs:
            logging.info(
                'Solution found with index "%s"', sol.index_name)
        else:
            logging.info('Solution not found')

