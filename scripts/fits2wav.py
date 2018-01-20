#!/usr/bin/env python

"""
Sonify a FITS image
"""

from __future__ import absolute_import, division, print_function

import os.path
import argparse
from glob import glob
import astropy.io.fits as pyfits
from skylib.sonification import sonify_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'infile', metavar='"infile"', nargs='?', default='*.fits',
        help='comma-separated list of input FITS file names/masks')
    parser.add_argument(
        '-a', '--coordinates', metavar='rect|radial|circ',
        choices=('rect', 'radial', 'circ'),
        default='rect', help='sonification coordinates')
    parser.add_argument(
        '-b', '--barycenter', action='store_true',
        help='move origin to barycenter')
    parser.add_argument(
        '-c', '--min-connected', metavar='n', type=int, default=5,
        help='minimum number of pixels per connected group')
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='write intermediate images')
    parser.add_argument(
        '-i', '--noise-hi', metavar='%', type=float, default=99.9,
        help='hi percentile noise clipping value')
    parser.add_argument(
        '-l', '--noise-lo', metavar='%', type=float, default=50.0,
        help='lo percentile noise clipping value')
    parser.add_argument(
        '-m', '--sampling-rate', metavar='n', type=int, default=44100,
        help='sampling rate in beats per second')
    parser.add_argument(
        '-n', '--noise-volume', metavar='n', type=int, default=1000,
        help='max noise volume')
    parser.add_argument(
        '-o', '--tones', metavar='n', type=int, default=22,
        help='number of discrete tones')
    parser.add_argument(
        '-p', '--hi-clip', metavar='%', type=float, default=99.9,
        help='high percentile image clipping value')
    parser.add_argument(
        '-r', '--threshold', metavar='x', type=float,
        default=1.5, help='detection threshold in units of RMS')
    parser.add_argument(
        '-s', '--background-scale', metavar='n|f', type=float, default=1/64,
        help='background inhomogeneity scale, pixels or fraction of image size')
    parser.add_argument(
        '-t', '--tempo', metavar='x', type=float, default=100.0,
        help='rows per second')
    parser.add_argument(
        '-v', '--volume', metavar='n', type=int, default=16384,
        help='max sound volume')
    args = parser.parse_args()

    for infile in sum([glob(mask) for mask in args.infile.split(',')], []):
        print('Generating sound from image "{}"'.format(infile))
        with pyfits.open(infile, 'readonly', uint=True) as f:
            img = f[0].data.astype(float)
        outfile = os.path.splitext(infile)[0] + '_{}.wav'.format(args.coord)
        print('Writing waveform file "{}"'.format(outfile))
        sonify_image(
            img, outfile, coord=args.coord, barycenter=args.barycenter,
            tempo=args.tempo, sampling_rate=args.sampling_rate,
            num_tones=args.tones, volume=args.volume,
            noise_volume=args.noise_volume, bkg_scale=args.background_scale,
            threshold=args.threshold, min_connected=args.min_connected,
            hi_clip=args.hi_clip, noise_lo=args.noise_lo,
            noise_hi=args.noise_hi)


if __name__ == '__main__':
    main()
