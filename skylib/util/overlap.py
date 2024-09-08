"""
Licensed under a 3-clause BSD style license.

Functions for calculating exact overlap between shapes.

Original cython version by Thomas Robitaille. Converted to C by Kyle Barbary.
Ported to Numba by Vladimir Kouprianov.
"""

import numpy as np
from numba import njit


__all__ = ['circoverlap', 'ellipoverlap', 'njitc']


def njitc(*args, **kws):
    """
    Equivalent to njit(..., nogil=True, cache=True, error_model='numpy')
    """
    kws.setdefault('nogil', True)
    kws.setdefault('cache', True)
    kws.setdefault('error_model', 'numpy')
    return njit(*args, **kws)


@njitc
def area_arc(x1: float, y1: float, x2: float, y2: float, r: float) -> float:
    """
    Return area of a circle arc between (x1, y1) and (x2, y2) with radius r
    reference: https://mathworld.wolfram.com/CircularSegment.html
    """
    if r <= 0:
        return 0.0
    a = np.hypot(x2 - x1, y2 - y1)
    theta = 2*np.arcsin(0.5*a/r)
    return 0.5*r**2*(theta - np.sin(theta))


@njitc
def area_triangle(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    """
    Area of a triangle defined by three vertices
    """
    return 0.5*abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))


@njitc
def circoverlap_core(xmin: float, ymin: float, xmax: float, ymax: float, r: float) -> float:
    """
    Core of circular overlap routine.
    Assumes that xmax >= xmin >= 0.0, ymax >= ymin >= 0.0. (can always modify input to conform to this).
    """
    xmin2 = xmin**2
    ymin2 = ymin**2
    r2 = r**2
    if xmin2 + ymin2 > r2:
        return 0

    xmax2 = xmax**2
    ymax2 = ymax**2
    if xmax2 + ymax2 < r2:
        return (xmax - xmin)*(ymax - ymin)

    a = xmax2 + ymin2  # (corner 1 distance)^2
    b = xmin2 + ymax2  # (corner 2 distance)^2

    if a < r2 and b < r2:
        x1 = np.sqrt(r2 - ymax2)
        y1 = ymax
        x2 = xmax
        y2 = np.sqrt(r2 - xmax2)
        return ((xmax-xmin)*(ymax-ymin) -
                area_triangle(x1, y1, x2, y2, xmax, ymax) +
                area_arc(x1, y1, x2, y2, r))

    if a < r2:
        x1 = xmin
        y1 = np.sqrt(r2 - xmin2)
        x2 = xmax
        y2 = np.sqrt(r2 - xmax2)
        return (area_arc(x1, y1, x2, y2, r) +
                area_triangle(x1, y1, x1, ymin, xmax, ymin) +
                area_triangle(x1, y1, x2, ymin, x2, y2))

    if b < r2:
        x1 = np.sqrt(r2 - ymin2)
        y1 = ymin
        x2 = np.sqrt(r2 - ymax2)
        y2 = ymax
        return (area_arc(x1, y1, x2, y2, r) +
                area_triangle(x1, y1, xmin, y1, xmin, ymax) +
                area_triangle(x1, y1, xmin, y2, x2, y2))

    x1 = np.sqrt(r2 - ymin2)
    y1 = ymin
    x2 = xmin
    y2 = np.sqrt(r2 - xmin2)
    return area_arc(x1, y1, x2, y2, r) + area_triangle(x1, y1, x2, y2, xmin, ymin)


@njitc
def circoverlap(xmin: float, ymin: float, xmax: float, ymax: float, r: float) -> float:
    """
    Area of overlap of a rectangle and a circle
    """
    # some subroutines demand that r > 0
    if r <= 0:
        return 0

    if xmin >= 0:
        if ymin >= 0:
            return circoverlap_core(xmin, ymin, xmax, ymax, r)
        if ymax <= 0:
            return circoverlap_core(xmin, -ymax, xmax, -ymin, r)
        return circoverlap_core(xmin, 0, xmax, -ymin, r) + circoverlap_core(xmin, 0, xmax, ymax, r)

    if xmax <= 0:
        if ymin >= 0:
            return circoverlap_core(-xmax, ymin, -xmin, ymax, r)
        if ymax <= 0:
            return circoverlap_core(-xmax, -ymax, -xmin, -ymin, r)
        return circoverlap_core(-xmax, 0, -xmin, -ymin, r) + circoverlap_core(-xmax, 0, -xmin, ymax, r)

    if ymin >= 0:
        return circoverlap_core(0, ymin, -xmin, ymax, r) + circoverlap_core(0, ymin, xmax, ymax, r)
    if ymax <= 0:
        return circoverlap_core(0, -ymax, -xmin, -ymin, r) + circoverlap_core(0, -ymax, xmax, -ymin, r)
    return (circoverlap_core(0, 0, -xmin, -ymin, r) +
            circoverlap_core(0, 0, xmax, -ymin, r) +
            circoverlap_core(0, 0, -xmin, ymax, r) +
            circoverlap_core(0, 0, xmax, ymax, r))


# *****************************************************************************/
# ellipse overlap functions

@njitc
def in_triangle(x: float, y: float, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> bool:
    """
    Check if a point (x,y) is inside a triangle
    """
    return (
        (((y1 > y) ^ (y2 > y)) and x - x1 < (x2 - x1)*(y - y1)/(y2 - y1)) ^
        (((y2 > y) ^ (y3 > y)) and x - x2 < (x3 - x2)*(y - y2)/(y3 - y2)) ^
        (((y3 > y) ^ (y1 > y)) and x - x3 < (x1 - x3)*(y - y3)/(y1 - y3))
    )


@njitc
def circle_line(x1: float, y1: float, x2: float, y2: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Intersection of a line defined by two points with a unit circle
    """
    tol = 1e-10

    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) < tol and abs(dy) < tol:
        return (2, 2), (2, 2)

    if abs(dx) > abs(dy):
        # Find the slope and intercept of the line
        a = dy/dx
        b = y1 - a*x1

        # Find the determinant of the quadratic equation
        delta = 1 + a**2 - b**2
        if delta > 0:  # solutions exist
            delta = np.sqrt(delta)
            p1x = (-a*b - delta)/(1 + a**2)
            p2x = (-a*b + delta) / (1 + a*a)
            return (p1x, a*p1x + b), (p2x, a*p2x + b)
        # no solution, return values > 1
        return (2, 2), (2, 2)

    # Find the slope and intercept of the line
    a = dx/dy
    b = x1 - a*y1

    # Find the determinant of the quadratic equation
    delta = 1 + a**2 - b**2
    if delta > 0:  # solutions exist
        delta = np.sqrt(delta)
        p1y = (-a*b - delta)/(1 + a**2)
        p2y = (-a*b + delta)/(1 + a**2)
        return (a*p1y + b, p1y), (a*p2y + b, p2y)

    # no solution, return values > 1
    return (2, 2), (2, 2)


@njitc
def circle_segment_single2(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """
    The intersection of a line with the unit circle. The intersection the closest to (x2, y2) is chosen.
    """
    inter = circle_line(x1, y1, x2, y2)
    pt1 = inter[0]
    pt2 = inter[1]

    # Can be optimized, but just checking for correctness right now
    dx1 = abs(pt1[0] - x2)
    dy1 = abs(pt1[1] - y2)
    dx2 = abs(pt2[0] - x2)
    dy2 = abs(pt2[1] - y2)

    if dx1 > dy1:  # compare based on x-axis
        if dx1 > dx2:
            return pt2
        return pt1
    if dy1 > dy2:
        return pt2
    return pt1


@njitc
def circle_segment(x1: float, y1: float, x2: float, y2: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Intersection(s) of a segment with the unit circle. Discard any solution not on the segment.
    """
    (pt1x, pt1y), (pt2x, pt2y) = circle_line(x1, y1, x2, y2)

    if pt1x > x1 and pt1x > x2 or pt1x < x1 and pt1x < x2 or \
            pt1y > y1 and pt1y > y2 or pt1y < y1 and pt1y < y2:
        pt1x = pt1y = 2
    if pt2x > x1 and pt2x > x2 or pt2x < x1 and pt2x < x2 or \
            pt2y > y1 and pt2y > y2 or pt2y < y1 and pt2y < y2:
        pt2x = pt2y = 2

    if pt1x > 1 and pt2x < 2:
        return (pt1x, pt1y), (pt2x, pt2y)
    return (pt2x, pt2y), (pt1x, pt1y)


@njitc
def triangle_unitcircle_overlap(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    """
    Given a triangle defined by three points (x1, y1), (x2, y2), and
    (x3, y3), find the area of overlap with the unit circle.
    """
    # Find distance of all vertices to circle center
    d1 = x1**2 + y1**2
    d2 = x2**2 + y2**2
    d3 = x3**2 + y3**2

    # Order vertices by distance from origin
    if d1 < d2:
        if d2 < d3:
            pass
        elif d1 < d3:
            x2, x3 = x3, x2
            y2, y3 = y3, y2
            d2, d3 = d3, d2
        else:
            x1, x3, x2 = x3, x2, x1
            y1, y3, y2 = y3, y2, y1
            d1, d3, d2 = d3, d2, d1
    else:
        if d1 < d3:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            d1, d2 = d2, d1
        elif d2 < d3:
            x1, x2, x3 = x2, x3, x1
            y1, y2, y3 = y2, y3, y1
            d1, d2, d3 = d2, d3, d1
        else:
            x1, x3 = x3, x1
            y1, y3 = y3, y1
            d1, d3 = d3, d1

    # Determine number of vertices inside circle
    in1 = d1 < 1
    in2 = d2 < 1
    in3 = d3 < 1

    # Determine which vertices are on the circle
    on1 = abs(d1 - 1) < 1e-10
    on2 = abs(d2 - 1) < 1e-10
    on3 = abs(d3 - 1) < 1e-10

    if on3 or in3:  # triangle completely within circle
        return area_triangle(x1, y1, x2, y2, x3, y3)

    if in2 or on2:
        # If vertex 1 or 2 are on the edge of the circle, then we use the dot product to vertex 3 to determine whether
        # an intersection takes place.
        intersect13 = not on1 or x1*(x3 - x1) + y1*(y3 - y1) < 0
        intersect23 = not on2 or x2*(x3 - x2) + y2*(y3 - y2) < 0
        if intersect13 and intersect23:
            pt1x, pt1y = circle_segment_single2(x1, y1, x3, y3)
            pt2x, pt2y = circle_segment_single2(x2, y2, x3, y3)
            return (area_triangle(x1, y1, x2, y2, pt1x, pt1y) +
                    area_triangle(x2, y2, pt1x, pt1y, pt2x, pt2y) +
                    area_arc(pt1x, pt1y, pt2x, pt2y, 1))
        if intersect13:
            pt1x, pt1y = circle_segment_single2(x1, y1, x3, y3)
            return area_triangle(x1, y1, x2, y2, pt1x, pt1y) + area_arc(x2, y2, pt1x, pt1y, 1)
        if intersect23:
            pt2x, pt2y = circle_segment_single2(x2, y2, x3, y3)
            return area_triangle(x1, y1, x2, y2, pt2x, pt2y) + area_arc(x1, y1, pt2x, pt2y, 1)
        return area_arc(x1, y1, x2, y2, 1)

    if in1:
        # Check for intersections of far side with circle
        (pt1x, pt1y), (pt2x, pt2y) = circle_segment(x2, y2, x3, y3)
        pt3x, pt3y = circle_segment_single2(x1, y1, x2, y2)
        pt4x, pt4y = circle_segment_single2(x1, y1, x3, y3)

        if pt1x > 1:  # indicates no intersection
            # check if the pixel vertex (x1, y2) and the origin are on
            # different sides of the circle segment. If they are, the
            # circle segment spans more than pi radians.
            # We use the formula (y-y1) * (x2-x1) > (y2-y1) * (x-x1)
            # to determine if (x, y) is on the left of the directed
            # line segment from (x1, y1) to (x2, y2)
            if ((-pt3y*(pt4x - pt3x) > -pt3x*(pt4y - pt3y)) !=
                    ((y1 - pt3y)*(pt4x - pt3x) > (x1 - pt3x)*(pt4y - pt3y))):
                return area_triangle(x1, y1, pt3x, pt3y, pt4x, pt4y) + np.pi - area_arc(pt3x, pt3y, pt4x, pt4y, 1)
            return area_triangle(x1, y1, pt3x, pt3y, pt4x, pt4y) + area_arc(pt3x, pt3y, pt4x, pt4y, 1)

        # ensure that pt1 is the point closest to (x2, y2)
        if (pt2x - x2)**2 + (pt2y - y2)**2 < (pt1x - x2)**2 + (pt1y - y2)**2:
            (pt1x, pt1y), (pt2x, pt2y) = (pt2x, pt2y), (pt1x, pt1y)

        return (area_triangle(x1, y1, pt3x, pt3y, pt1x, pt1y) +
                area_triangle(x1, y1, pt1x, pt1y, pt2x, pt2y) +
                area_triangle(x1, y1, pt2x, pt2y, pt4x, pt4y) +
                area_arc(pt1x, pt1y, pt3x, pt3y, 1) +
                area_arc(pt2x, pt2y, pt4x, pt4y, 1))

    (pt1x, pt1y), (pt2x, pt2y) = circle_segment(x1, y1, x2, y2)
    if pt1x <= 1:
        xp = 0.5*(pt1x + pt2x)
        yp = 0.5*(pt1y + pt2y)
        return (triangle_unitcircle_overlap(x1, y1, x3, y3, xp, yp) +
                triangle_unitcircle_overlap(x2, y2, x3, y3, xp, yp))

    (pt1x, pt1y), (pt2x, pt2y) = circle_segment(x2, y2, x3, y3)
    if pt1x <= 1:
        xp = 0.5*(pt1x + pt2x)
        yp = 0.5*(pt1y + pt2y)
        return (triangle_unitcircle_overlap(x3, y3, x1, y1, xp, yp) +
                triangle_unitcircle_overlap(x2, y2, x1, y1, xp, yp))

    (pt1x, pt1y), _ = circle_segment(x3, y3, x1, y1)
    if pt1x <= 1:
        xp = 0.5*(pt1x + pt2x)
        yp = 0.5*(pt1y + pt2y)
        return (triangle_unitcircle_overlap(x1, y1, x2, y2, xp, yp) +
                triangle_unitcircle_overlap(x3, y3, x2, y2, xp, yp))

    # no intersections
    if in_triangle(0, 0, x1, y1, x2, y2, x3, y3):
        return np.pi
    return 0


@njitc
def ellipoverlap(xmin: float, ymin: float, xmax: float, ymax: float, a: float, b: float, theta: float) -> float:
    """
    Exact overlap between a rectangle defined by (xmin, ymin, xmax, ymax) and an ellipse with major and minor axes
    rx and ry respectively and position angle theta.
    """
    cos_m_theta = np.cos(-theta)
    sin_m_theta = np.sin(-theta)

    # scale by which the areas will be shrunk
    scale = a*b

    # Reproject rectangle to a frame in which ellipse is a unit circle
    x1 = (xmin*cos_m_theta - ymin*sin_m_theta)/a
    y1 = (xmin*sin_m_theta + ymin*cos_m_theta)/b
    x2 = (xmax*cos_m_theta - ymin*sin_m_theta)/a
    y2 = (xmax*sin_m_theta + ymin*cos_m_theta)/b
    x3 = (xmax*cos_m_theta - ymax*sin_m_theta)/a
    y3 = (xmax*sin_m_theta + ymax*cos_m_theta)/b
    x4 = (xmin*cos_m_theta - ymax*sin_m_theta)/a
    y4 = (xmin*sin_m_theta + ymax*cos_m_theta)/b

    # Divide resulting quadrilateral into two triangles and find intersection with unit circle
    return scale*(triangle_unitcircle_overlap(x1, y1, x2, y2, x3, y3) +
                  triangle_unitcircle_overlap(x1, y1, x4, y4, x3, y3))
