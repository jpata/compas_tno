from numpy import array
from numpy import ones
from numpy import zeros
from numpy import concatenate
from numpy import linspace

from compas_tno.shapes import rectangular_topology
from compas_tno.shapes import MeshDos

import math


def cross_vault_highfields_proxy(xy_span=[[0.0, 10.0], [0.0, 10.0]], thk=0.5, tol=10e-6, t=0.0, discretisation=[100, 100], *args, **kwargs):
    """Function that computes the highfield of cross vaults through the proxy

    Parameters
    ----------
    xy_span : [list[float], list[float]], optional
        xy-span of the shape, by default [[0.0, 10.0], [0.0, 10.0]]
    thk : float, optional
        The thickness of the vault, by default 0.5
    tol : float, optional
        Tolerance, by default 10e-6
    t : float, optional
        Parameter for lower bound in nodes in the boundary, by default 0.0
    discretisation : list|int, optional
        Level of discretisation of the shape, by default [100, 100]

    Returns
    -------
    intradosdata
        Data to a Mesh for the intrados of the shape
    extradosdata
        Data to a Mesh for the extrados of the shape
    middledata
        Data to a Mesh for the middle of the shape
    """
    intrados, extrados, middle = cross_vault_highfields(xy_span=xy_span, thk=thk, tol=tol, t=t, discretisation=discretisation, expanded=False)
    return intrados.to_data(), extrados.to_data(), middle.to_data()


def cross_vault_highfields(xy_span=[[0.0, 10.0], [0.0, 10.0]], thk=0.50, tol=10e-6, t=0.0, discretisation=[100, 100], expanded=False):
    """ Set Cross-Vault heights.

    Parameters
    ----------
    xy_span : [list[float], list[float]], optional
        xy-span of the shape, by default [[0.0, 10.0], [0.0, 10.0]]
    thk : float, optional
        The thickness of the vault, by default 0.5
    tol : float, optional
        Tolerance, by default 10e-6
    t : float, optional
        Parameter for lower bound in nodes in the boundary, by default 0.0
    discretisation : list|int, optional
        Level of discretisation of the shape, by default [100, 100]

    Returns
    -------
    intrados : :class:`~compas_tno.shapes.MeshDos`
        A MeshDos for the intrados of the shape
    extrados : :class:`~compas_tno.shapes.MeshDos`
        A MeshDos for the extrados of the shape
    middle : :class:`~compas_tno.shapes.MeshDos`
        A MeshDos for the middle of the shape

    Notes
    ----------------------
    Position of the quadrants is as in the schema below:

        Q3
    Q2      Q1
        Q4

    """

    if isinstance(discretisation, int):
        discretisation = [discretisation, discretisation]

    y1 = xy_span[1][1]
    y0 = xy_span[1][0]
    x1 = xy_span[0][1]
    x0 = xy_span[0][0]

    density_x = discretisation[0]
    density_y = discretisation[1]
    x = linspace(x0, x1, num=density_x+1, endpoint=True)  # arange(x0, x1 + dx/density_x, dx/density_x)
    y = linspace(y0, y1, num=density_y+1, endpoint=True)  # arange(y0, y1 + dy/density_y, dy/density_y)

    xi, yi, faces_i = rectangular_topology(x, y)

    zt = crossvault_middle_update(xi, yi, t, xy_span=xy_span, tol=1e-6)
    xyzt = array([xi, yi, zt.flatten()]).transpose()
    middle = MeshDos.from_vertices_and_faces(xyzt, faces_i)

    if expanded:
        x = concatenate(([x0 - thk/2], x, [x1 + thk/2]))
        y = concatenate(([y0 - thk/2], y, [y1 + thk/2]))
        xi, yi, faces_i = rectangular_topology(x, y)
        zub, zlb = crossvault_ub_lb_update(xi, yi, thk, t, xy_span=xy_span, tol=1e-6)
    else:
        zub, zlb = crossvault_ub_lb_update(xi, yi, thk, t, xy_span=xy_span, tol=1e-6)

    xyzub = array([xi, yi, zub.flatten()]).transpose()
    xyzlb = array([xi, yi, zlb.flatten()]).transpose()

    extrados = MeshDos.from_vertices_and_faces(xyzub, faces_i)
    intrados = MeshDos.from_vertices_and_faces(xyzlb, faces_i)

    return intrados, extrados, middle


def crossvault_ub_lb_update(x, y, thk, t, xy_span=[[0.0, 10.0], [0.0, 10.0]], tol=1e-6):
    """Update upper and lower bounds of an crossvault based in the parameters

    Parameters
    ----------
    x : list
        x-coordinates of the points
    y : list
        y-coordinates of the points
    thk : float
        Thickness of the arch
    t : float
        Parameter for lower bound in nodes in the boundary
    xy_span : [list[float], list[float]], optional
        xy-span of the shape, by default [[0.0, 10.0], [0.0, 10.0]]
    tol : float, optional
        Tolerance, by default 10e-6

    Returns
    -------
    ub : array
        Values of the upper bound in the points
    lb : array
        Values of the lower bound in the points
    """

    y1 = xy_span[1][1]
    y0 = xy_span[1][0]
    x1 = xy_span[0][1]
    x0 = xy_span[0][0]

    y1_ub = y1 + thk/2
    y0_ub = y0 - thk/2
    x1_ub = x1 + thk/2
    x0_ub = x0 - thk/2

    y1_lb = y1 - thk/2
    y0_lb = y0 + thk/2
    x1_lb = x1 - thk/2
    x0_lb = x0 + thk/2

    rx_ub = (x1_ub - x0_ub)/2
    ry_ub = (y1_ub - y0_ub)/2
    rx_lb = (x1_lb - x0_lb)/2
    ry_lb = (y1_lb - y0_lb)/2

    hc_ub = max(rx_ub, ry_ub)
    hc_lb = max(rx_lb, ry_lb)

    ub = ones((len(x), 1))
    lb = ones((len(x), 1)) * - t

    for i in range(len(x)):
        xi, yi = x[i], y[i]
        xd_ub = x0_ub + (x1_ub - x0_ub)/(y1_ub - y0_ub) * (yi - y0_ub)
        yd_ub = y0_ub + (y1_ub - y0_ub)/(x1_ub - x0_ub) * (xi - x0_ub)
        hxd_ub = math.sqrt((rx_ub)**2 - ((xd_ub - x0_ub) - rx_ub)**2)
        hyd_ub = math.sqrt((ry_ub)**2 - ((yd_ub - y0_ub) - ry_ub)**2)

        intrados_null = False

        if (yi > y1_lb and (xi > x1_lb or xi < x0_lb)) or (yi < y0_lb and (xi > x1_lb or xi < x0_lb)):
            intrados_null = True
        else:
            yi_intra = yi
            xi_intra = xi
            if yi > y1_lb:
                yi_intra = y1_lb
            if yi < y0_lb:
                yi_intra = y0_lb
            elif xi > x1_lb:
                xi_intra = x1_lb
            elif xi < x0_lb:
                xi_intra = x0_lb

            xd_lb = x0_lb + (x1_lb - x0_lb)/(y1_lb - y0_lb) * (yi_intra - y0_lb)
            yd_lb = y0_lb + (y1_lb - y0_lb)/(x1_lb - x0_lb) * (xi_intra - x0_lb)
            hxd_lb = _sqrt(((rx_lb)**2 - ((xd_lb - x0_lb) - rx_lb)**2))
            hyd_lb = _sqrt(((ry_lb)**2 - ((yd_lb - y0_lb) - ry_lb)**2))

        if yi <= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) + tol and yi >= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) - tol:  # Q1
            ub[i] = hc_ub*(hxd_ub + math.sqrt((ry_ub)**2 - ((yi - y0_ub) - ry_ub)**2))/(rx_ub + ry_ub)
            if not intrados_null:
                lb[i] = hc_lb*(hxd_lb + math.sqrt((ry_lb)**2 - ((yi_intra - y0_lb) - ry_lb)**2))/(rx_lb + ry_lb)
        elif yi >= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) - tol and yi >= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) - tol:  # Q3
            ub[i] = hc_ub*(hyd_ub + math.sqrt((rx_ub)**2 - ((xi - x0_ub) - rx_ub)**2))/(rx_ub + ry_ub)
            if not intrados_null:
                lb[i] = hc_lb*(hyd_lb + math.sqrt((rx_lb)**2 - ((xi_intra - x0_lb) - rx_lb)**2))/(rx_lb + ry_lb)
        elif yi >= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) - tol and yi <= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) + tol:  # Q2
            ub[i] = hc_ub*(hxd_ub + math.sqrt((ry_ub)**2 - ((yi - y0_ub) - ry_ub)**2))/(rx_ub + ry_ub)
            if not intrados_null:
                lb[i] = hc_lb*(hxd_lb + math.sqrt((ry_lb)**2 - ((yi_intra - y0_lb) - ry_lb)**2))/(rx_lb + ry_lb)
        elif yi <= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) + tol and yi <= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) + tol:  # Q4
            ub[i] = hc_ub*(hyd_ub + math.sqrt((rx_ub)**2 - ((xi - x0_ub) - rx_ub)**2))/(rx_ub + ry_ub)
            if not intrados_null:
                lb[i] = hc_lb*(hyd_lb + math.sqrt((rx_lb)**2 - ((xi_intra - x0_lb) - rx_lb)**2))/(rx_lb + ry_lb)
        else:
            print('Error Q. (x,y) = ({0},{1})'.format(xi, yi))

    return ub, lb


def crossvault_dub_dlb(x, y, thk, t, xy_span=[[0.0, 10.0], [0.0, 10.0]], tol=1e-6):
    """Computes the sensitivities of upper and lower bounds in the x, y coordinates and thickness specified.

    Parameters
    ----------
    x : list
        x-coordinates of the points
    y : list
        y-coordinates of the points
    thk : float
        Thickness of the arch
    t : float
        Parameter for lower bound in nodes in the boundary
    xy_span : [list[float], list[float]], optional
        xy-span of the shape, by default [[0.0, 10.0], [0.0, 10.0]]
    tol : float, optional
        Tolerance, by default 10e-6

    Returns
    -------
    dub : array
        Values of the sensitivities for the upper bound in the points
    dlb : array
        Values of the sensitivities for the lower bound in the points
    """

    y1 = xy_span[1][1]
    y0 = xy_span[1][0]
    x1 = xy_span[0][1]
    x0 = xy_span[0][0]

    y1_ub = y1 + thk/2
    y0_ub = y0 - thk/2
    x1_ub = x1 + thk/2
    x0_ub = x0 - thk/2

    y1_lb = y1 - thk/2
    y0_lb = y0 + thk/2
    x1_lb = x1 - thk/2
    x0_lb = x0 + thk/2

    rx_ub = (x1_ub - x0_ub)/2
    ry_ub = (y1_ub - y0_ub)/2
    rx_lb = (x1_lb - x0_lb)/2
    ry_lb = (y1_lb - y0_lb)/2

    hc_ub = max(rx_ub, ry_ub)
    hc_lb = max(rx_lb, ry_lb)

    ub = ones((len(x), 1))
    lb = ones((len(x), 1)) * - t
    dub = zeros((len(x), 1))  # dzub / dt
    dlb = zeros((len(x), 1))  # dzlb / dt

    dubdx = zeros((len(x), len(x)))
    dubdy = zeros((len(x), len(x)))
    dlbdx = zeros((len(x), len(x)))
    dlbdy = zeros((len(x), len(x)))

    yc = ry_ub + y0_ub  # Only works for square
    xc = rx_ub + x0_ub

    for i in range(len(x)):

        xi, yi = x[i], y[i]
        xd_ub = x0_ub + (x1_ub - x0_ub)/(y1_ub - y0_ub) * (yi - y0_ub)
        yd_ub = y0_ub + (y1_ub - y0_ub)/(x1_ub - x0_ub) * (xi - x0_ub)
        hxd_ub = math.sqrt((rx_ub)**2 - ((xd_ub - x0_ub) - rx_ub)**2)
        hyd_ub = math.sqrt((ry_ub)**2 - ((yd_ub - y0_ub) - ry_ub)**2)

        intrados_null = False

        if (yi > y1_lb and (xi > x1_lb or xi < x0_lb)) or (yi < y0_lb and (xi > x1_lb or xi < x0_lb)):
            intrados_null = True
        else:
            yi_intra = yi
            xi_intra = xi
            if yi > y1_lb:
                yi_intra = y1_lb
            elif yi < y0_lb:
                yi_intra = y0_lb
            if xi > x1_lb:
                xi_intra = x1_lb
            elif xi < x0_lb:
                xi_intra = x0_lb

            xd_lb = x0_lb + (x1_lb - x0_lb)/(y1_lb - y0_lb) * (yi_intra - y0_lb)
            yd_lb = y0_lb + (y1_lb - y0_lb)/(x1_lb - x0_lb) * (xi_intra - x0_lb)
            hxd_lb = _sqrt(((rx_lb)**2 - ((xd_lb - x0_lb) - rx_lb)**2))
            hyd_lb = _sqrt(((ry_lb)**2 - ((yd_lb - y0_lb) - ry_lb)**2))

        if yi <= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) + tol and yi >= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) - tol:  # Q1
            ub[i] = hc_ub*(hxd_ub + math.sqrt((ry_ub)**2 - ((yi - y0_ub) - ry_ub)**2))/(rx_ub + ry_ub)
            dub[i] = 1/2 * ry_ub/ub[i] * hc_ub/((rx_ub + ry_ub)/2)
            # dubdx[i, i] += 0.0
            dubdy[i, i] += - (yi - yc) / ub[i]
            if not intrados_null:
                lb[i] = hc_lb*(hxd_lb + math.sqrt((ry_lb)**2 - ((yi_intra - y0_lb) - ry_lb)**2))/(rx_lb + ry_lb)
                dlb[i] = - 1/2 * ry_lb/lb[i] * hc_lb/((rx_lb + ry_lb)/2)
                # dlbdx[i, i] += 0.0
                dlbdy[i, i] += - (yi - yc) / lb[i]
        if yi >= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) - tol and yi >= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) - tol:  # Q3
            ub[i] = hc_ub*(hyd_ub + math.sqrt((rx_ub)**2 - ((xi - x0_ub) - rx_ub)**2))/(rx_ub + ry_ub)
            dub[i] = 1/2 * rx_ub/ub[i] * hc_ub/((rx_ub + ry_ub)/2)
            # dubdy[i, i] += 0.0
            dubdx[i, i] += - (xi - xc) / ub[i]
            if not intrados_null:
                lb[i] = hc_lb*(hyd_lb + math.sqrt((rx_lb)**2 - ((xi_intra - x0_lb) - rx_lb)**2))/(rx_lb + ry_lb)
                dlb[i] = - 1/2 * rx_lb/lb[i] * hc_lb/((rx_lb + ry_lb)/2)
                # dlbdy[i, i] += 0.0
                dlbdx[i, i] += - (xi - xc) / lb[i]
        if yi >= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) - tol and yi <= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) + tol:  # Q2
            ub[i] = hc_ub*(hxd_ub + math.sqrt((ry_ub)**2 - ((yi - y0_ub) - ry_ub)**2))/(rx_ub + ry_ub)
            dub[i] = 1/2 * ry_ub/ub[i] * hc_ub/((rx_ub + ry_ub)/2)
            # dubdx[i, i] += 0.0
            dubdy[i, i] += - (yi - yc) / ub[i]
            if not intrados_null:
                lb[i] = hc_lb*(hxd_lb + math.sqrt((ry_lb)**2 - ((yi_intra - y0_lb) - ry_lb)**2))/(rx_lb + ry_lb)
                dlb[i] = - 1/2 * ry_lb/lb[i] * hc_lb/((rx_lb + ry_lb)/2)
                # dlbdx[i, i] += 0.0
                dlbdy[i, i] += - (yi - yc) / lb[i]
        if yi <= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) + tol and yi <= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) + tol:  # Q4
            ub[i] = hc_ub*(hyd_ub + math.sqrt((rx_ub)**2 - ((xi - x0_ub) - rx_ub)**2))/(rx_ub + ry_ub)
            dub[i] = 1/2 * rx_ub/ub[i] * hc_ub/((rx_ub + ry_ub)/2)
            # dubdy[i, i] += 0.0
            dubdx[i, i] += - (xi - xc) / ub[i]
            if not intrados_null:
                lb[i] = hc_lb*(hyd_lb + math.sqrt((rx_lb)**2 - ((xi_intra - x0_lb) - rx_lb)**2))/(rx_lb + ry_lb)
                dlb[i] = - 1/2 * rx_lb/lb[i] * hc_lb/((rx_lb + ry_lb)/2)
                # dlbdy[i, i] += 0.0
                dlbdx[i, i] += - (xi - xc) / lb[i]
        # else:
        #     print('Error Q. (x,y) = ({0},{1})'.format(xi, yi))

    return dub, dlb, dubdx, dubdy, dlbdx, dlbdy  # ub, lb


def crossvault_middle_update(x, y, t, xy_span=[[0.0, 10.0], [0.0, 10.0]], tol=1e-6):
    """Update middle of a crossvault based in the parameters

    Parameters
    ----------
    x : list
        x-coordinates of the points
    y : list
        y-coordinates of the points
    thk : float
        Thickness of the arch
    t : float
        Parameter for lower bound in nodes in the boundary
    xy_span : [list[float], list[float]], optional
        xy-span of the shape, by default [[0.0, 10.0], [0.0, 10.0]]
    tol : float, optional
        Tolerance, by default 10e-6

    Returns
    -------
    z : array
        Values of the middle surface in the points
    """

    y1 = xy_span[1][1]
    y0 = xy_span[1][0]
    x1 = xy_span[0][1]
    x0 = xy_span[0][0]

    rx = (x1 - x0)/2
    ry = (y1 - y0)/2
    hc = max(rx, ry)

    z = zeros((len(x), 1))

    for i in range(len(x)):
        xi, yi = x[i], y[i]
        if yi > y1:
            yi = y1
        if yi < y0:
            yi = y0
        if xi > x1:
            xi = x1
        if xi < x0:
            xi = x0
        xd = x0 + (x1 - x0)/(y1 - y0) * (yi - y0)
        yd = y0 + (y1 - y0)/(x1 - x0) * (xi - x0)
        hxd = math.sqrt(abs((rx)**2 - ((xd - x0) - rx)**2))
        hyd = math.sqrt(abs((ry)**2 - ((yd - y0) - ry)**2))
        if yi <= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) + tol and yi >= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) - tol:  # Q1
            z[i] = hc*(hxd + math.sqrt((ry)**2 - ((yi - y0) - ry)**2))/(rx + ry)
        elif yi >= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) - tol and yi >= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) - tol:  # Q3
            z[i] = hc*(hyd + math.sqrt((rx)**2 - ((xi - x0) - rx)**2))/(rx + ry)
        elif yi >= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) - tol and yi <= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) + tol:  # Q2
            z[i] = hc*(hxd + math.sqrt((ry)**2 - ((yi - y0) - ry)**2))/(rx + ry)
        elif yi <= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) + tol and yi <= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) + tol:  # Q4
            z[i] = hc*(hyd + math.sqrt((rx)**2 - ((xi - x0) - rx)**2))/(rx + ry)
        else:
            print('Vertex did not belong to any Q. (x,y) = ({0},{1})'.format(xi, yi))
            z[i] = -t
    return z


def _sqrt(x):
    try:
        sqrt_x = math.sqrt(x)
    except BaseException:
        if x > -10e4:
            sqrt_x = math.sqrt(abs(x))
        else:
            sqrt_x = 0.0
            print('Problems to sqrt: ', x)
    return sqrt_x


def _calculate_segmental_arch_z(coord, c0, span_c, rise_c, tol=1e-9):
    """
    Calculates z-coordinate of a segmental arch springing from z=0.

    Parameters
    ----------
    coord : float
        The x or y coordinate at which to calculate the height.
    c0 : float
        The starting x or y coordinate of the arch's span.
    span_c : float
        The total span of the arch.
    rise_c : float
        The maximum rise (height) of the arch at its center.
    tol : float, optional
        Tolerance for floating point comparisons, by default 1e-9.

    Returns
    -------
    float
        The z-coordinate (height) of the arch at the given coord.
    """
    if rise_c <= tol:  # Flat or no arch if rise is negligible
        return 0.0
    
    lc = span_c
    hc = rise_c
    
    # Radius of the segmental arch
    if abs(hc) < tol: # Avoid division by zero if hc is effectively zero (already handled, but good check)
        return 0.0
    # Formula for radius of a circular segment: R = h/2 + c^2/(8h) where c is chord length (span)
    radius_c = (hc / 2.0) + (lc**2 / (8.0 * hc))
    
    # Z-coordinate of the circle's center assuming arch springs from z=0
    # The lowest point of the circle is at z = z_center_c - radius_c
    # We want the arch to spring from z=0, so the lowest point of the arch part is 0.
    # z_coord_of_circle_center = -(radius_c - hc)
    z_coord_of_circle_center = hc - radius_c # if arch springs from z=0 and top is at hc

    # X-coordinate of the arch's center (mid-span)
    coord_arch_center = c0 + lc / 2.0
    
    dist_from_arch_center = abs(coord - coord_arch_center)
    
    if dist_from_arch_center > lc / 2.0 + tol: # Point is outside the span
        return 0.0 
        
    squared_val_for_sqrt = radius_c**2 - dist_from_arch_center**2
    
    if squared_val_for_sqrt < -tol : # Should ideally not happen if dist_from_arch_center <= lc/2
        return 0.0 
    elif squared_val_for_sqrt < 0: # very small negative due to precision, treat as zero
        squared_val_for_sqrt = 0.0

    # Height from the circle's center: sqrt(R^2 - x_dist^2)
    # Then adjust by the circle's center z-coordinate to place arch correctly
    # zc = math.sqrt(squared_val_for_sqrt) + z_coord_of_circle_center # This would put the crown at hc
    zc = math.sqrt(squared_val_for_sqrt) - (radius_c - hc)


    return max(0.0, zc) # Ensure z is not negative if springing from z=0


def flatter_crossvault_middle_update(x_coords, y_coords, t, xy_span, desired_max_rise, spr_angle=0.0, tol=1e-6):
    """
    Update z-coordinates for the middle surface of a flatter cross vault.
    Assumes springing from z=0 (spr_angle mostly ignored for simplicity here).
    """
    x0, x1 = xy_span[0]
    y0, y1 = xy_span[1]
    Lx = x1 - x0
    Ly = y1 - y0

    # Effective rise for each generating arch
    # The generating arch rises to `desired_max_rise` if its span allows, otherwise it's semi-circular up to `desired_max_rise`
    rise_x_arch = min(desired_max_rise, Lx / 2.0)
    rise_y_arch = min(desired_max_rise, Ly / 2.0)

    z_middle = zeros((len(x_coords), 1))

    # Calculate springing height adjustment if spr_angle is used
    # For simplicity, let's assume spr_angle means the entire vault base is lifted,
    # or it defines the starting tangent which is more complex for non-circular.
    # Here, we'll assume it implies the actual vault starts at z_spring_base.
    # This part might need refinement based on exact interpretation of spr_angle for cross vaults.
    z_spring_base_x = 0.0
    z_spring_base_y = 0.0
    if spr_angle > tol: # A simple interpretation: an inclined base affecting the rise calculation.
        # This is tricky for cross vaults. A common pavillion vault interpretation:
        # z_spring_base_x = (Lx / 2.0) * math.tan(math.radians(spr_angle))
        # z_spring_base_y = (Ly / 2.0) * math.tan(math.radians(spr_angle))
        pass # For now, assume flatter cross vaults spring from z=0 or handle t for boundary.


    for i in range(len(x_coords)):
        xi, yi = x_coords[i], y_coords[i]
        
        # Height contribution from the arch spanning X-direction
        zx = _calculate_segmental_arch_z(xi, x0, Lx, rise_x_arch, tol) + z_spring_base_x
        
        # Height contribution from the arch spanning Y-direction
        zy = _calculate_segmental_arch_z(yi, y0, Ly, rise_y_arch, tol) + z_spring_base_y
        
        z_middle[i] = max(zx, zy)
        
        # Ensure boundary conditions related to 't' are met (if vault edge is below z=0)
        # This logic is usually for lb, but middle surface shouldn't go below -t conceptually
        is_on_x_boundary = abs(xi - x0) < tol or abs(xi - x1) < tol
        is_on_y_boundary = abs(yi - y0) < tol or abs(yi - y1) < tol
        if is_on_x_boundary and is_on_y_boundary: # Corner
             z_middle[i] = max(z_middle[i], -t) # Should be 0 if t=0
        elif is_on_x_boundary or is_on_y_boundary: # Edge
             z_middle[i] = max(z_middle[i], -t) # Should be 0 if t=0


    return z_middle

def flatter_crossvault_ub_lb_update(x_coords, y_coords, thk, t, xy_span, desired_max_rise, spr_angle=0.0, tol=1e-6):
    """
    Update upper and lower bounds for a flatter cross vault.
    Uses a simplified vertical offset from the middle surface.
    """
    z_middle = flatter_crossvault_middle_update(x_coords, y_coords, t, xy_span, desired_max_rise, spr_angle, tol)
    
    ub = ones((len(x_coords), 1))
    lb = ones((len(x_coords), 1))

    for i in range(len(x_coords)):
        ub[i] = z_middle[i] + thk / 2.0
        # Ensure lower bound respects 't' and doesn't go below the conceptual ground plane
        # A more robust lb would ensure it's not > ub and handle springing properly
        potential_lb = z_middle[i] - thk / 2.0
        
        # Boundary check for 't' (simplified: if on boundary, lb can go down to -t)
        xi, yi = x_coords[i], y_coords[i]
        x0, x1 = xy_span[0]
        y0, y1 = xy_span[1]
        is_on_boundary = abs(xi - x0) < tol or abs(xi - x1) < tol or \
                         abs(yi - y0) < tol or abs(yi - y1) < tol
        
        if is_on_boundary:
            lb[i] = max(potential_lb, -t) 
        else:
            lb[i] = potential_lb
            
        # Ensure lb is not above z=0 if vault springs from ground, unless t allows negative
        if t <= tol:
            lb[i] = max(0.0, lb[i]) 
        
        # Final check: lb <= ub
        if lb[i] > ub[i] - tol: # Ensure a minimum thickness
            lb[i] = ub[i] - tol 


    return ub, lb

def flatter_crossvault_heightfields(xy_span=[[0.0, 10.0], [0.0, 10.0]], thk=0.50,
                                   desired_max_rise=None, t=0.0, discretisation=[100, 100],
                                   spr_angle=0.0, tol=1e-6): # expanded parameter removed for now
    """
    Generates intrados, extrados, and middle surface meshes for a flatter cross vault.
    """
    if isinstance(discretisation, int):
        discretisation = [discretisation, discretisation]

    y1 = xy_span[1][1]
    y0 = xy_span[1][0]
    x1 = xy_span[0][1]
    x0 = xy_span[0][0]

    density_x = discretisation[0]
    density_y = discretisation[1]
    # Generate a grid of points for the plan
    x_range = linspace(x0, x1, num=density_x + 1, endpoint=True)
    y_range = linspace(y0, y1, num=density_y + 1, endpoint=True)

    # Create the topology (xi, yi are flattened lists of coordinates, faces_i uses indices)
    xi, yi, faces_i = rectangular_topology(x_range, y_range)

    # Calculate z-coordinates for the middle surface
    zt_flat = flatter_crossvault_middle_update(xi, yi, t, xy_span, desired_max_rise, spr_angle, tol)
    xyzt = array([xi, yi, zt_flat.flatten()]).transpose()
    middle = MeshDos.from_vertices_and_faces(xyzt, faces_i)

    # Calculate z-coordinates for upper (extrados) and lower (intrados) bounds
    zub_flat, zlb_flat = flatter_crossvault_ub_lb_update(xi, yi, thk, t, xy_span, desired_max_rise, spr_angle, tol)
    
    xyzub = array([xi, yi, zub_flat.flatten()]).transpose()
    xyzlb = array([xi, yi, zlb_flat.flatten()]).transpose()

    extrados = MeshDos.from_vertices_and_faces(xyzub, faces_i)
    intrados = MeshDos.from_vertices_and_faces(xyzlb, faces_i)
    
    # expanded logic could be added here if needed, by extending x_range, y_range before surface calculation

    return intrados, extrados, middle


# def crossvault_dub_dlb_old(x, y, thk, t, xy_span=[[0.0, 10.0], [0.0, 10.0]], tol=1e-6):

#     y1 = xy_span[1][1]
#     y0 = xy_span[1][0]
#     x1 = xy_span[0][1]
#     x0 = xy_span[0][0]

#     y1_ub = y1 + thk/2
#     y0_ub = y0 - thk/2
#     x1_ub = x1 + thk/2
#     x0_ub = x0 - thk/2

#     y1_lb = y1 - thk/2
#     y0_lb = y0 + thk/2
#     x1_lb = x1 - thk/2
#     x0_lb = x0 + thk/2

#     rx_ub = (x1_ub - x0_ub)/2
#     ry_ub = (y1_ub - y0_ub)/2
#     rx_lb = (x1_lb - x0_lb)/2
#     ry_lb = (y1_lb - y0_lb)/2

#     hc_ub = max(rx_ub, ry_ub)
#     hc_lb = max(rx_lb, ry_lb)

#     ub = ones((len(x), 1))
#     lb = ones((len(x), 1)) * - t
#     dub = zeros((len(x), 1))
#     dlb = zeros((len(x), 1))

#     for i in range(len(x)):

#         xi, yi = x[i], y[i]
#         xd_ub = x0_ub + (x1_ub - x0_ub)/(y1_ub - y0_ub) * (yi - y0_ub)
#         yd_ub = y0_ub + (y1_ub - y0_ub)/(x1_ub - x0_ub) * (xi - x0_ub)
#         hxd_ub = math.sqrt((rx_ub)**2 - ((xd_ub - x0_ub) - rx_ub)**2)
#         hyd_ub = math.sqrt((ry_ub)**2 - ((yd_ub - y0_ub) - ry_ub)**2)

#         intrados_null = False

#         if (yi > y1_lb and (xi > x1_lb or xi < x0_lb)) or (yi < y0_lb and (xi > x1_lb or xi < x0_lb)):
#             intrados_null = True
#         else:
#             yi_intra = yi
#             xi_intra = xi
#             if yi > y1_lb:
#                 yi_intra = y1_lb
#             elif yi < y0_lb:
#                 yi_intra = y0_lb
#             if xi > x1_lb:
#                 xi_intra = x1_lb
#             elif xi < x0_lb:
#                 xi_intra = x0_lb

#             xd_lb = x0_lb + (x1_lb - x0_lb)/(y1_lb - y0_lb) * (yi_intra - y0_lb)
#             yd_lb = y0_lb + (y1_lb - y0_lb)/(x1_lb - x0_lb) * (xi_intra - x0_lb)
#             hxd_lb = _sqrt(((rx_lb)**2 - ((xd_lb - x0_lb) - rx_lb)**2))
#             hyd_lb = _sqrt(((ry_lb)**2 - ((yd_lb - y0_lb) - ry_lb)**2))

#         if yi <= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) + tol and yi >= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) - tol:  # Q1
#             ub[i] = hc_ub*(hxd_ub + math.sqrt((ry_ub)**2 - ((yi - y0_ub) - ry_ub)**2))/(rx_ub + ry_ub)
#             dub[i] = 1/2 * ry_ub/ub[i] * hc_ub/((rx_ub + ry_ub)/2)
#             if not intrados_null:
#                 lb[i] = hc_lb*(hxd_lb + math.sqrt((ry_lb)**2 - ((yi_intra - y0_lb) - ry_lb)**2))/(rx_lb + ry_lb)
#                 dlb[i] = - 1/2 * ry_lb/lb[i] * hc_lb/((rx_lb + ry_lb)/2)
#         elif yi >= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) - tol and yi >= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) - tol:  # Q3
#             ub[i] = hc_ub*(hyd_ub + math.sqrt((rx_ub)**2 - ((xi - x0_ub) - rx_ub)**2))/(rx_ub + ry_ub)
#             dub[i] = 1/2 * rx_ub/ub[i] * hc_ub/((rx_ub + ry_ub)/2)
#             if not intrados_null:
#                 lb[i] = hc_lb*(hyd_lb + math.sqrt((rx_lb)**2 - ((xi_intra - x0_lb) - rx_lb)**2))/(rx_lb + ry_lb)
#                 dlb[i] = - 1/2 * rx_lb/lb[i] * hc_lb/((rx_lb + ry_lb)/2)
#         elif yi >= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) - tol and yi <= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) + tol:  # Q2
#             ub[i] = hc_ub*(hxd_ub + math.sqrt((ry_ub)**2 - ((yi - y0_ub) - ry_ub)**2))/(rx_ub + ry_ub)
#             dub[i] = 1/2 * ry_ub/ub[i] * hc_ub/((rx_ub + ry_ub)/2)
#             if not intrados_null:
#                 lb[i] = hc_lb*(hxd_lb + math.sqrt((ry_lb)**2 - ((yi_intra - y0_lb) - ry_lb)**2))/(rx_lb + ry_lb)
#                 dlb[i] = - 1/2 * ry_lb/lb[i] * hc_lb/((rx_lb + ry_lb)/2)
#         elif yi <= y0 + (y1 - y0)/(x1 - x0) * (xi - x0) + tol and yi <= y1 - (y1 - y0)/(x1 - x0) * (xi - x0) + tol:  # Q4
#             ub[i] = hc_ub*(hyd_ub + math.sqrt((rx_ub)**2 - ((xi - x0_ub) - rx_ub)**2))/(rx_ub + ry_ub)
#             dub[i] = 1/2 * rx_ub/ub[i] * hc_ub/((rx_ub + ry_ub)/2)
#             if not intrados_null:
#                 lb[i] = hc_lb*(hyd_lb + math.sqrt((rx_lb)**2 - ((xi_intra - x0_lb) - rx_lb)**2))/(rx_lb + ry_lb)
#                 dlb[i] = - 1/2 * rx_lb/lb[i] * hc_lb/((rx_lb + ry_lb)/2)
#         else:
#             print('Error Q. (x,y) = ({0},{1})'.format(xi, yi))

#     return dub, dlb  # ub, lb


# def crossvault_ub_lb_update_repetitive(x, y, thk, t, xy_span=[[0.0, 10.0], [0.0, 10.0]], tol=1e-6):  # Invoke the method crossvault middle many times

#     ub = ones((len(x), 1))
#     lb = ones((len(x), 1)) * - t

#     rx = (xy_span[0][1] - xy_span[0][0])/2
#     ry = (xy_span[1][1] - xy_span[1][0])/2

#     if ry > rx:
#         y1_ub = xy_span[1][1] + thk/2 * rx/ry
#         y0_ub = xy_span[1][0] - thk/2 * rx/ry
#         x1_ub = xy_span[0][1] + thk/2
#         x0_ub = xy_span[0][0] - thk/2
#         xy_span_ub = [[x0_ub, x1_ub], [y0_ub, y1_ub]]

#         y1_lb = xy_span[1][1] - thk/2 * rx/ry
#         y0_lb = xy_span[1][0] + thk/2 * rx/ry
#         x1_lb = xy_span[0][1] - thk/2
#         x0_lb = xy_span[0][0] + thk/2
#         xy_span_lb = [[x0_lb, x1_lb], [y0_lb, y1_lb]]
#     else:
#         y1_ub = xy_span[1][1] + thk/2
#         y0_ub = xy_span[1][0] - thk/2
#         x1_ub = xy_span[0][1] + thk/2 * ry/rx
#         x0_ub = xy_span[0][0] - thk/2 * ry/rx
#         xy_span_ub = [[x0_ub, x1_ub], [y0_ub, y1_ub]]

#         y1_lb = xy_span[1][1] - thk/2
#         y0_lb = xy_span[1][0] + thk/2
#         x1_lb = xy_span[0][1] - thk/2 * ry/rx
#         x0_lb = xy_span[0][0] + thk/2 * ry/rx
#         xy_span_lb = [[x0_lb, x1_lb], [y0_lb, y1_lb]]

#     ub = crossvault_middle_update(x, y, t, xy_span=xy_span_ub)
#     lb = crossvault_middle_update(x, y, t, xy_span=xy_span_lb)

#     return ub, lb
