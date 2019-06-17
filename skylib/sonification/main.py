"""
Implementation of the sonification algorithm

sonify_image(): generate a WAV file from image data.
to_polar(): transform image to radial or circular coordinates.
"""

from __future__ import absolute_import, division, print_function

import os
import wave

from numpy import (
    arange, array, ceil, cos, indices, int16, percentile, pi, sin, sqrt, outer,
    zeros)
from numpy.random import normal
from scipy.ndimage import (
    find_objects, generate_binary_structure, label, map_coordinates, shift)
from scipy.interpolate import interp1d

from ..calibration.background import estimate_background


__all__ = ['sonify_image']


def to_polar(img, noise_map, coord):
    """
    Transform image (and, optionally, noise map) into polar coordinates

    Output is a 2D image with X corresponding to radius and Y to angle (coord =
    "radial") or vice versa (coord = "circ").

    :param array_like img: input image
    :param array_like | None noise_map: optional noise map, same shape as `img`
    :param str coord: coordinate type: "radial" or "circ"

    :return: transformed image and noise map; shape is (h x w) for circular
        mapping and (w x h) for radial mapping, where (h x w) is the input image
        shape
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    h, w = img.shape
    idx = indices([h, w], float).ravel().reshape([2, w*h])
    r, theta = idx.copy()
    r *= sqrt(2)/2
    theta *= 2*pi/w
    x, y = w/2 + r*cos(theta), h/2 + r*sin(theta)
    c = array([y, x])
    img = map_coordinates(img, c).reshape([h, w])
    if noise_map is not None:
        noise_map = map_coordinates(noise_map, c).reshape([h, w])

    # Expand r axis to remove the possible large empty areas for large r
    img[abs(img) < 1e-7] = 0
    try:
        max_r = img.nonzero()[0].max()
    except ValueError:
        max_r = 0
    # noinspection PyTypeChecker
    if 0 < max_r < 0.9*h:
        c = array([idx[0]*(max_r/h), idx[1]])
        img = map_coordinates(img, c).reshape([h, w])
        if noise_map is not None:
            noise_map = map_coordinates(noise_map, c).reshape([h, w])

    if coord == 'radial':
        # Swap r and theta for a radial sweep
        img = img.T
        if noise_map is not None:
            noise_map = noise_map.T

    return img, noise_map


def sonify_image(img, outfile, coord='rect', barycenter=False, tempo=100.0,
                 sampling_rate=44100, start_tone=0, num_tones=22, volume=16384,
                 noise_volume=1000, bkg_scale=1/64, threshold=1.5,
                 min_connected=5, hi_clip=99.9, noise_lo=50.0, noise_hi=99.9,
                 bkg=None, rms=None, index_sounds=False):
    """
    Transform an image to sound and write it to a WAV file

    :param array_like img: 2D input image
    :param str | file-like outfile: output WAV file name
    :param str coord: type of coordinates: "rect", "radial", or "circ"
    :param bool barycenter: shift origin to barycenter
    :param float tempo: rows per second for the output file
    :param int sampling_rate: output file sampling rate
    :param int start_tone: tone to start the scale from; 0 = C4, 1 = C#4, -1 =
        B3, etc.
    :param int num_tones: number of major scale tones to use
    :param int volume: output volume scaling (0 to 32767)
    :param int noise_volume: noise volume scaling (0 to 32767)
    :param int | float bkg_scale: box size for background estimation: either
        an integer value in pixels or a floating-point value from 0 to 1 in
        units of image size
    :param float threshold: detection threshold in units of noise RMS
    :param int min_connected: minimum number of connected pixels above threshold
        for object detection
    :param float hi_clip: high image data clipping percentile (0 to 100)
    :param float noise_lo: low noise clipping percentile (0 to 100)
    :param float noise_hi: high noise clipping percentile (0 to 100)
    :param array_like bkg: optional background map, same shape as `img`; if not
        supplied, the background and RMS are estimated from the image; makes
        sense when sonifying a small part of the whole image, where the
        background from the given subimage may be overestimated for bright
        extended sources
    :param array_like rms: optional background RMS map, same shape as `img`;
        must be supplied along with `background`
    :param bool index_sounds: enable start and stop index sounds

    :rtype: None
    """
    # Normalize parameters (could be strings when invoked from a web app)
    barycenter = bool(int(barycenter))
    tempo = float(tempo)
    sampling_rate = int(sampling_rate)
    num_tones = int(num_tones)
    volume = int(volume)
    noise_volume = int(noise_volume)
    if not isinstance(bkg_scale, float):
        try:
            bkg_scale = int(bkg_scale)
        except ValueError:
            bkg_scale = float(bkg_scale)
    threshold = float(threshold)
    min_connected = int(min_connected)
    hi_clip = float(hi_clip)
    noise_lo = float(noise_lo)
    noise_hi = float(noise_hi)

    # Subtract background and clip pixels below detection threshold
    h, w = img.shape
    if bkg is None or rms is None:
        bkg, rms = estimate_background(img, size=bkg_scale)
    if noise_volume:
        noise_map = rms
    else:
        noise_map = None
    img -= bkg + rms*threshold
    # noinspection PyTypeChecker
    img = img.clip(0, percentile(img, hi_clip))

    # Mask pixels not belonging to connected groups
    labels, n = label(img, generate_binary_structure(2, 1))
    for s in find_objects(labels, n):
        if len(labels[s].nonzero()[0]) < min_connected:
            img[s] = 0
    try:
        img -= img[(img > 0).nonzero()].min()
    except ValueError:
        # No pixels above zero
        pass

    if barycenter:
        # Shift origin to barycenter
        y, x = indices([h, w])
        s = img.sum()
        dx, dy = w/2 - (x*img).sum()/s, h/2 - (y*img).sum()/s
        x_plus, y_plus = int(ceil(abs(dx))), int(ceil(abs(dy)))
        if dx < 0:
            ofs_x = x_plus
        else:
            ofs_x = 0
        if dy < 0:
            ofs_y = y_plus
        else:
            ofs_y = 0
        temp_img = zeros([h + y_plus, w + x_plus])
        temp_img[ofs_y:ofs_y + h, ofs_x:ofs_x + w] = img
        img = shift(temp_img, (dy, dx))
        # noinspection PyTypeChecker
        img[abs(img) < 1e-7] = 0
        if noise_map is not None:
            temp_img[:] = 0
            temp_img[ofs_y:ofs_y + h, ofs_x:ofs_x + w] = noise_map
            noise_map = shift(temp_img, (dy, dx))
            # noinspection PyTypeChecker
            noise_map[abs(noise_map) < 1e-7] = 0
        h, w = img.shape

    if coord in ('radial', 'circ'):
        img, noise_map = to_polar(img, noise_map, coord)
        h, w = img.shape

    # Split the image into vertical bands corresponding to C major scale notes
    c_major = array([0, 2, 2, 1, 2, 2, 2]).cumsum()
    n = num_tones
    tones = arange(n)
    scale = (2*pi*440)*2**(tones//7 + (c_major[tones % 7] + start_tone - 9)/12)
    v = zeros([h + 1, n])
    bw = w//n
    for i in range(n):
        v[:-1, i] = img[:, i*bw:(i + 1)*bw].sum(1)

    # Normalize volumes and build a linear interpolator
    max_vol = v.max()
    if max_vol:
        v *= volume/max_vol
    v = interp1d(arange(h + 1)/tempo, v, axis=0)

    # Calculate noise volume
    v_noise = None
    if noise_map is not None:
        # noinspection PyTypeChecker
        min_noise, max_noise = percentile(noise_map, [noise_lo, noise_hi])
        noise_range = max_noise - min_noise
        if noise_range:
            noise_map = (noise_map - min_noise).clip(0, noise_range)
            v_noise = zeros([h + 1, n])
            for i in range(n):
                v_noise[:-1, i] = noise_map[:, i*bw:(i + 1)*bw].sum(1)
            max_vol = v_noise.max()
            if max_vol:
                v_noise *= noise_volume/max_vol
            v_noise = interp1d(arange(h + 1)/tempo, v_noise, axis=0)

    # Generate stereo waveform for each tone, modulated by the volume linearly
    # interpolated between rows
    samples_per_row = sampling_rate/tempo
    x = arange(n, dtype=float)
    x /= n - 1
    pan = array([1 - x, x]).T  # LR pan weights
    noise_pan = pan/pan.sum()

    ofs = 0
    outfile = wave.open(outfile, 'wb')
    try:
        outfile.setnchannels(2)
        outfile.setsampwidth(2)
        outfile.setframerate(sampling_rate)

        index_sound_dir = os.path.realpath(os.path.join(
            os.getcwd(), os.path.dirname(__file__)))
        if index_sounds:
            si = wave.open(os.path.join(index_sound_dir, 'start.wav'), 'rb')
            try:
                outfile.writeframes(si.readframes(si.getnframes()))
            finally:
                si.close()

        for i in range(h):
            # Number of samples per the current image row; works also for
            # unevenly spaced rows (fractional samples_per_row)
            m = int((i + 1)*samples_per_row + 0.5) - ofs

            # Continuous time in seconds starting from the top of the image
            t = (i + arange(m)/m)/tempo

            # Scale by volume (linearly interpolated for each t), do a linear
            # pan with scale[0] -> left, scale[-1] -> right
            waveform = ((v(t)*sin(outer(t, scale)))[..., None]*pan).sum(1)

            # Generate white noise
            if noise_volume is not None and v_noise is not None:
                waveform += ((v_noise(t)*normal(
                    0, 1, (m, n)))[..., None]*noise_pan).sum(1)

            # Clip to 16-bit signed int and write to WAV
            outfile.writeframes(waveform.clip(
                -2**15, 2**15 - 1).astype(int16).tostring())

            ofs += m

        if index_sounds:
            ei = wave.open(os.path.join(index_sound_dir, 'stop.wav'), 'rb')
            try:
                outfile.writeframes(ei.readframes(ei.getnframes()))
            finally:
                ei.close()
    finally:
        outfile.close()
