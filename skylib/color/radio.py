"""
Radio-specific multi-wavelength transformations
"""

import numpy as np


__all__ = ['radio_nat']


def radio_nat(img_r, img_g, img_b, fcal_r, fcal_g, fcal_b, ftrue_r, ftrue_g,
              ftrue_b, nu_r, nu_g, nu_b, lambda_r=630, lambda_g=532,
              lambda_b=465, betamax=3, floor=0):
    img_r = img_r.astype(float)
    img_g = img_g.astype(float)
    img_b = img_b.astype(float)

    # Transform G first
    if floor:
        fg = img_g + floor
    else:
        fg = img_g.copy()

    # Transform R and B
    for img, fcal, ftrue, nu, lam in [
            (img_r, fcal_r, ftrue_r, nu_r, lambda_r),
            (img_b, fcal_b, ftrue_b, nu_b, lambda_b)]:
        # First we need to rescale the source map values -- putting everything
        # on G's instrumental efficiency level
        log_nu = np.log(nu_g/nu)
        img *= fcal_g/fcal*(nu/nu_g)**(np.log(ftrue_g/ftrue)/log_nu)
        img += floor

        # Now that they match the efficiency, we now need to scale the images,
        # so they aren't just white
        a = (np.log(fg/img)/log_nu).clip(-betamax, betamax)

        # Handle NaNs at fg = img = 0 and fg/img < 0
        a[np.isnan(a).nonzero()] = 0

        good = ((img > 0) & (fg > 0)).nonzero()
        img[good] = (fg*(lambda_g/lam)**a)[good]
        img -= floor

    for img in (img_r, img_g, img_b):
        img *= ftrue_g/fcal_g

    return img_r, img_g, img_b
