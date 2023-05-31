import numpy as np
import cv2
import math
import time
import random

from typing import List

#simulation parameters
Vt = 36
Vr = 72
Vs = 23
Rmax = 200
Vc = 40
Phi = 1.0

width, height = 1001, 1001 #pixels
canvas = np.zeros((height, width, 3), np.uint8)
settings_canvas = np.zeros((100, 300, 3), np.uint8)

sim_width, sim_height = 2000, 2000
grid_ratio = sim_width / width
grid_scale = 25 #meters

pattern_canvas = np.zeros((grid_scale * 2, width, 3), np.uint8)

grid_width = 0
grid_height = 0

vec_grid = []

pattern = []


def compute_rankine(x, y, Vr, Vt, Vs, Rmax, Phi):

    r = math.hypot(x, y)

    radial_unit_vec = [-x / r, -y / r]

    tangential_unit_vec = [radial_unit_vec[1], -radial_unit_vec[0]]

    scale_factor = (r * (1 / Rmax)) ** Phi if r <= Rmax else ((1 / r) * Rmax) ** Phi

    Vtan = Vt * scale_factor
    Vrad = Vr * scale_factor

    tangential_vec = [tangential_unit_vec[0] * Vtan, tangential_unit_vec[1] * Vtan]

    radial_vec = [radial_unit_vec[0] * Vrad, radial_unit_vec[1] * Vrad]

    return [tangential_vec[0] + radial_vec[0], tangential_vec[1] + radial_vec[1] + Vs]


def compute_rankine_unit(x, y, Vr, Vt, Vs, Rmax, Phi):
    vec = compute_rankine(x, y, Vr, Vt, Vs, Rmax, Phi)

    mag = math.hypot(vec[0], vec[1])

    return [vec[0] / mag, vec[1] / mag]


def solve_vmax(Vr, Vt, Vs):
    return Vs + math.sqrt(Vr * Vr + Vt * Vt)


def solve_asymptotes(Vr, Vt, Vs, Vc, Rmax):
    Vr2 = Vr * Vr
    Vt2 = Vt * Vt
    Vs2 = Vs * Vs
    Vc2 = Vc * Vc

    r2t2 = Vr2 + Vt2
    s2c2 = Vs2 - Vc2
    Rt = Rmax * Vt
    Rr = Rmax * Vr

    d = r2t2 * (2 * (Vs2 + Vc2) - r2t2) - (s2c2 * s2c2)

    if d < 0:
        return [float('nan'), float('nan'), float('nan'), float('nan')]

    k1 = -Rt * (r2t2 + s2c2)
    k2 = Rr * math.sqrt(d)
    k3 = 1.0 / (2 * Vs * r2t2)

    asym1 = (k1 + k2) * k3
    asym2 = (k1 - k2) * k3

    cs = Vc + Vs
    c_s = Vc - Vs

    k4 = Rr * (r2t2 + s2c2)
    k5 = Rt * math.sqrt(-(r2t2 - c_s*c_s)*(r2t2 - cs*cs))

    yasym1 = (k4 + k5) * k3
    yasym2 = (k4 - k5) * k3

    return [asym2, asym1, yasym2, yasym1]


def solve_convergence(ellipse_params: tuple, Vr, Vt, outer: bool):
    A, B, C, D, E, F = ellipse_params

    k1 = Vr * E - Vt * D
    k2 = Vr * Vr * C - Vt * (Vr * B - Vt * A)
    s = -math.copysign(1.0, C) if outer else math.copysign(1.0, C)

    return Vt * ((k1 + s * math.sqrt(k1*k1 - 4*F*k2)) / (2*k2))


def solve_pattern_width(ellipse_params: tuple):
    A, B, C, D, E, F = ellipse_params

    AC = A * C
    AE = A * E
    BE = B * E
    CD = C * D
    B2 = B * B

    k1 = 2*CD - BE
    k2 = B2 - 4*AC
    k2_1 = 1/k2
    k3 = 2*math.sqrt(C*(AE*E + F*k2 + D*(CD - BE)))

    x1 = (k1 + k3) * k2_1
    x2 = (k1 - k3) * k2_1

    y0 = (2*AE - B*D) * k2_1

    return [x1, x2, x2 - x1, y0]


def solve_ellipse_approx(pts: List[List[float]]):
    x1, y1 = pts[0][0], pts[0][1]
    x2, y2 = pts[1][0], pts[1][1]
    x3, y3 = pts[2][0], pts[2][1]
    x4, y4 = pts[3][0], pts[3][1]
    x5, y5 = pts[4][0], pts[4][1]

    m21 = y2 / (x2 - x1)
    m32 = y2 / (x2 - x3)
    m43 = y4 / (x4 - x3)
    m14 = y4 / (x4 - x1)

    x1x5 = x1 - x5
    x3x5 = x3 - x5

    L = -((y5 + m21*x1x5)*(y5 + m43*x3x5)) / ((y5 + m14*x1x5)*(y5 + m32*x3x5))

    A = m21*m43 + L*m14*m32
    B1 = m21 + L*m14
    B2 = m43 + L*m32
    B = B1 + B2
    C = 1 + L
    D = -A*(x1 + x3)
    E = B1*x1 + B2*x3
    F = A*x1*x3

    return A, B, C, D, E, F


def compute_anchor_pts(Vr, Vt, Vs, Vc, Rmax, Phi, outer: bool):

    pts = []

    Vr2 = Vr * Vr
    Vt2 = Vt * Vt
    Vs2 = Vs * Vs
    Vc2 = Vc * Vc
    p_1 = 1 / Phi
    R2p = Rmax * 2**(-p_1)
    sqrt2 = math.sqrt(2.0)

    k1 = (Vr2 + Vt2) * (2*Vc2 - Vs2)
    k2 = -Vs2 * (Vr2 - Vt2)
    k3 = -2*Vr*Vt*Vs2
    k4 = -2*Vr*Vs if outer else 2*Vr*Vs
    k5 = 2*Vt*Vs if outer else -2*Vt*Vs
    k6 = 1 / (Vc2 - Vs2) if outer else 1 / (Vr2 + Vt2)

    th2 = -0.00222222 * Vr + 0.00406163 * Vt + 4.05822 if outer else 0.843299 - 0.00203702 * Vr + 0.00404527 * Vt
    th4 = -0.0035 * Vr + 0.00760781 * Vt + 5.69782 if outer else 2.53383 - 0.00205015 * Vr + 0.00779394 * Vt
    th5 = 0.00255556 * Vr + 0.00406626 * Vt + 0.492544 if outer else 3.63039 + 0.00146659 * Vr + 0.00932935 * Vt

    thetas = [0, th2, math.pi, th4, th5]

    for th in thetas:
        sin = math.sin(th)
        cos = math.cos(th)
        sin2 = 2*sin*cos
        cos2 = cos * cos - sin * sin

        r = R2p * ((sqrt2*math.sqrt(k1 + k2*cos2 + k3*sin2) + k4*sin + k5*cos) * k6)**p_1

        pts.append([r*cos, r*sin])

    return pts


def solve_rankine_approx(x, ellipse_params: tuple, outer: bool):
    A, B, C, D, E, F = ellipse_params

    k1 = math.copysign(1.0, C) if outer else -math.copysign(1.0, C)
    k2 = B*x + E

    return (-k2 + k1 * math.sqrt(k2*k2 - 4*C*(x*(A*x + D) + F))) / (2*C)


def generate_pattern(dx, Vr, Vt, Vs, Vc, Rmax, Phi):
    #solve asymptotes
    asymptotes = solve_asymptotes(Vr, Vt, Vs, Vc, Rmax)

    # solve outer ellipse approximation coefficients
    outer_anchor_pts = compute_anchor_pts(Vr, Vt, Vs, Vc, Rmax, Phi, True)
    outer_ellipse_params = solve_ellipse_approx(outer_anchor_pts)

    pattern = []
    Xc = 0

    #if no asymptotes
    if math.isnan(asymptotes[1]):
        # solve convergence axis
        Xc = solve_convergence(outer_ellipse_params, Vr, Vt, True)

        # solve pattern widths
        w = solve_pattern_width(outer_ellipse_params)

        # compute pattern
        for x in range(int(Xc - ((Xc - w[0]) // dx)*dx), int(Xc + ((w[1] - Xc) // dx) * dx), int(dx)):
            y = solve_rankine_approx(x, outer_ellipse_params, True)
            vec = compute_rankine_unit(x, y, Vr, Vt, Vs, Rmax, Phi)

            pattern.append([x, y, vec[0], vec[1]])

    else:
        #solve inner ellipse approximation coefficients
        inner_anchor_pts = compute_anchor_pts(Vr, Vt, Vs, Vc, Rmax, Phi, False)
        inner_ellipse_params = solve_ellipse_approx(inner_anchor_pts)

        # solve pattern widths
        w = solve_pattern_width(outer_ellipse_params)

        #if only inner solutions exists
        Xc = solve_convergence(inner_ellipse_params, Vr, Vt, False) if (asymptotes[2] < w[3] and asymptotes[3] < w[3]) else asymptotes[1]

        # compute pattern
        if asymptotes[2] > w[3]:
            offset = int(Xc - ((Xc - w[0]) // dx)*dx)
            #outer
            for x in range(int(Xc - ((Xc - w[0]) // dx)*dx), int(asymptotes[0]), int(dx)):
                y = solve_rankine_approx(x, outer_ellipse_params, True)
                vec = compute_rankine_unit(x, y, Vr, Vt, Vs, Rmax, Phi)

                pattern.append([x, y, vec[0], vec[1]])
                offset = x

            offset1 = int(offset + dx)
            #inner
            for x in range(int(offset + dx), int(asymptotes[1]), int(dx)):
                y = solve_rankine_approx(x, inner_ellipse_params, False)
                vec = compute_rankine_unit(x, y, Vr, Vt, Vs, Rmax, Phi)

                pattern.append([x, y, vec[0], vec[1]])
                offset1 = x

            #outer
            for x in range(int(offset1 + dx), int(Xc + ((w[1] - Xc) // dx)*dx), int(dx)):
                y = solve_rankine_approx(x, outer_ellipse_params, True)
                vec = compute_rankine_unit(x, y, Vr, Vt, Vs, Rmax, Phi)

                pattern.append([x, y, vec[0], vec[1]])

        else:
            offset = int(Xc - ((Xc - asymptotes[0]) // dx)*dx)

            # inner
            for x in range(int(Xc - ((Xc - asymptotes[0]) // dx)*dx), int(asymptotes[1]), int(dx)):
                y = solve_rankine_approx(x, inner_ellipse_params, False)
                vec = compute_rankine_unit(x, y, Vr, Vt, Vs, Rmax, Phi)

                pattern.append([x, y, vec[0], vec[1]])
                offset = x

            if asymptotes[3] > w[3]:
                # outer
                for x in range(int(offset + dx), int(Xc + ((w[1] - Xc) // dx) * dx), int(dx)):
                    y = solve_rankine_approx(x, outer_ellipse_params, True)
                    vec = compute_rankine_unit(x, y, Vr, Vt, Vs, Rmax, Phi)

                    pattern.append([x, y, vec[0], vec[1]])

    return pattern, Xc


def generate_field(origin):

    for i in range(grid_height):
        for j in range(grid_width):

            if i == origin[1] and j == origin[0]:
                vec_grid[i][j] = [0, 0]
                continue

            radial_distance_vec = [(origin[0] - j) * grid_scale, (i - origin[1]) * grid_scale]

            r = math.hypot(radial_distance_vec[0], radial_distance_vec[1])

            radial_unit_vec = [radial_distance_vec[0] / r, radial_distance_vec[1] / r]

            tangential_unit_vec = [radial_unit_vec[1], -radial_unit_vec[0]]

            scale_factor = (r / Rmax)**Phi if r <= Rmax else (Rmax / r)**Phi

            Vtan = Vt * scale_factor
            Vrad = Vr * scale_factor

            tangential_vec = [tangential_unit_vec[0] * Vtan, tangential_unit_vec[1] * Vtan]

            radial_vec = [radial_unit_vec[0] * Vrad, radial_unit_vec[1] * Vrad]

            vec_grid[i][j] = [tangential_vec[0] + radial_vec[0], tangential_vec[1] + radial_vec[1] + Vs]

    # if 0 <= origin[0] < grid_height and 0 <= origin[1] < grid_width:
    #     sum = [0, 0]
    #     count = 0
    #     for i in range(max(origin[1] - 1, 0), min(origin[1] + 2, grid_height)):
    #         for j in range(max(origin[0] - 1, 0), min(origin[0] + 2, grid_width)):
    #             if i == origin[1] and j == origin[0]:
    #                 continue
    #             sum[0] += vec_grid[i][j][0]
    #             sum[1] += vec_grid[i][j][1]
    #             count += 1.0
    #
    #     vec_grid[origin[1]][origin[0]] = [sum[0]/count, sum[1]/count]


# def generate_pattern(origin):
#     for i in range(grid_width):
#         m = -9999999
#         w = -9999999
#         no_solution_count = 0
#         x = (i - origin[0]) * grid_scale
#
#         Rmax2 = Rmax * Rmax
#         x2 = x * x
#         Vt2 = Vt * Vt
#         Vr2 = Vr * Vr
#         Vs2 = Vs * Vs
#         Vcrit2 = Vcrit * Vcrit
#
#         minY = Rmax2 - x2
#
#         # r <= Rmax
#         a = (1 / Rmax2) * (Vr2 + Vt2)
#         b = -2 * (1 / Rmax) * Vr * Vs
#         c = a * x2 + 2 * (1 / Rmax) * Vt * Vs * x + Vs2 - Vcrit2
#
#         d = b*b - 4*a*c
#
#         if -0.001 <= c <= 0.001:
#             m = -b / a
#         elif d >= 0: #solution exists (not imaginary)
#             sqrtd = math.sqrt(d)
#
#             y1 = (-b + sqrtd) / (2 * a)
#             y2 = (-b - sqrtd) / (2 * a)
#
#             if y1 * y1 > minY:
#                 y1 = -99999999
#
#             if y2 * y2 > minY:
#                 y2 = -99999999
#
#             m = max(y1, y2)
#         else:
#             no_solution_count += 1
#
#         #r > Rmax
#         a = Rmax2 * (Vr2 + Vt2)
#         b = 2 * Rmax * Vr * Vs
#         c = 2 * Rmax * Vt * Vs
#         d = Vs2 - Vcrit2
#
#         q = b*b - 4*d*(a + x*(c + d * x))
#
#         if -0.001 <= d <= 0.001:
#             w = (a + c * x) / b
#
#         elif q >= 0: #solution exists (not imaginary)
#             sqrtq = math.sqrt(q)
#
#             y1 = (b + sqrtq) / (2 * d)
#             y2 = (b - sqrtq) / (2 * d)
#
#             if y1 * y1 <= minY:
#                 y1 = -99999999
#
#             if y2 * y2 <= minY:
#                 y2 = -99999999
#
#             w = max(y1, y2)
#         else:
#             no_solution_count += 1
#
#         if no_solution_count != 2:
#             y = max(w, m)
#
#             radial_distance_vec = [-x, -y]
#
#             r = math.hypot(radial_distance_vec[0], radial_distance_vec[1])
#
#             radial_unit_vec = [radial_distance_vec[0] / r, radial_distance_vec[1] / r]
#
#             tangential_unit_vec = [radial_unit_vec[1], -radial_unit_vec[0]]
#
#             scale_factor = r * (1 / Rmax) if m > w else (1 / r) * Rmax
#
#             Vtan = Vt * scale_factor
#             Vrad = Vr * scale_factor
#
#             tangential_vec = [tangential_unit_vec[0] * Vtan, tangential_unit_vec[1] * Vtan]
#
#             radial_vec = [radial_unit_vec[0] * Vrad, radial_unit_vec[1] * Vrad]
#
#             velocity = [tangential_vec[0] + radial_vec[0], tangential_vec[1] + radial_vec[1] + Vs]
#
#             mag = math.hypot(velocity[0], velocity[1])
#
#             if Vcrit * 1.05 >= mag >= Vcrit * 0.95:
#                 pattern[i] = [x, y, velocity[0] / mag, velocity[1] / mag]


#convert magnitude into color gradient
def mag_to_color(mag):
    color_delta = 15.0

    #BGR color cause opencv...
    colors = [[100, 0, 0], [255, 0, 0], [255, 128, 0], [225, 255, 20], [128, 255, 128], [20, 255, 225], [0, 145, 255], [0, 0, 255], [0, 0, 100]]

    norm_mag = mag / color_delta
    idx = int(math.floor(norm_mag))

    if idx < 8:
        rem = norm_mag - idx

        return [colors[idx + 1][0] * rem + colors[idx][0] * (1 - rem),
                colors[idx + 1][1] * rem + colors[idx][1] * (1 - rem),
                colors[idx + 1][2] * rem + colors[idx][2] * (1 - rem)]

    else:
        return colors[8]


def render_field_mag():

    #color magnitudes
    for i in range(grid_height):
        for j in range(grid_width):
            mag = math.hypot(vec_grid[i][j][0], vec_grid[i][j][1])

            gsf = grid_scale/grid_ratio
            cv2.rectangle(canvas, (int(j*gsf), int(i*gsf)), (int(j*gsf + gsf), int(i*gsf + gsf)), mag_to_color(mag), -1)


def render_field_arrows():
    #vector arrows
    for i in range(grid_height):
        for j in range(grid_width):
            mag = math.hypot(vec_grid[i][j][0], vec_grid[i][j][1])

            if mag == 0:
                continue

            gsf = grid_scale / grid_ratio

            scaled_vec = [vec_grid[i][j][0] * gsf * 0.75 / mag, vec_grid[i][j][1] * gsf * 0.75 / mag]

            center = [j*gsf + gsf/2, i*gsf + gsf/2]
            arrow_points = [[int(round(center[0] - scaled_vec[0]/2)), int(round(center[1] + scaled_vec[1]/2))],
                            [int(round(center[0] + scaled_vec[0]/2)), int(round(center[1] - scaled_vec[1]/2))]]

            cv2.arrowedLine(canvas, arrow_points[0], arrow_points[1], [0, 0, 0], 1, line_type=cv2.LINE_AA, tipLength=0.5)


# def render_pattern(origin):
#     for p in pattern:
#
#         if p == -1:
#             continue
#
#         gsf = grid_scale / grid_ratio
#         scaled_vec = [p[2] * gsf * 0.75, p[3] * gsf * 0.75]
#
#         center = [(p[0] + origin[0]*grid_scale) / grid_ratio, (origin[1]*grid_scale - p[1]) / grid_ratio]
#
#         arrow_points = [[int(round(center[0] - scaled_vec[0] / 2)), int(round(center[1] + scaled_vec[1] / 2))],
#                         [int(round(center[0] + scaled_vec[0] / 2)), int(round(center[1] - scaled_vec[1] / 2))]]
#
#         cv2.arrowedLine(canvas, arrow_points[0], arrow_points[1], [0, 0, 0], 1, line_type=cv2.LINE_AA, tipLength=0.5)
#
#         scaled_vec = [p[2] * grid_scale, p[3] * grid_scale]
#         center = [(p[0] + origin[0] * grid_scale) / grid_ratio * 2, pattern_canvas.shape[0] / 2.0]
#
#         arrow_points = [[int(round(center[0] - scaled_vec[0] / 2)), int(round(center[1] + scaled_vec[1] / 2))],
#                         [int(round(center[0] + scaled_vec[0] / 2)), int(round(center[1] - scaled_vec[1] / 2))]]
#
#         cv2.arrowedLine(pattern_canvas, arrow_points[0], arrow_points[1], [255], 1, line_type=cv2.LINE_AA, tipLength=0.4)


def render_pattern(pattern, Xc, origin):
    #convergence = RankineFastLib.solveConvergenceRankine(Vt, Vr, Vs, Vcrit, Rmax)

    cv2.line(pattern_canvas, [int(round((Xc + sim_width / 2) / grid_ratio)), 0], [int(round((Xc + sim_width / 2) / grid_ratio)), pattern_canvas.shape[0]], [0, 0, 255], 1)

    #w = RankineFastLib.solvePatternWidth(Vt, Vr, Vs, Vcrit, Rmax)

    #print(w[0], w[1], w[0] - w[1])

    #cv2.line(pattern_canvas, [int(round((w[0] + sim_width / 2) / grid_ratio)), 0], [int(round((w[0] + sim_width / 2) / grid_ratio)), pattern_canvas.shape[0]], [0, 0, 255], 1)
    #cv2.line(pattern_canvas, [int(round((w[1] + sim_width / 2) / grid_ratio)), 0], [int(round((w[1] + sim_width / 2) / grid_ratio)), pattern_canvas.shape[0]], [0, 0, 255], 1)

    for p in pattern:

        # if p[1] == 0 and p[2] == 0 and p[3] == 0:
        #     continue

        gsf = grid_scale / grid_ratio
        scaled_vec = [p[2] * gsf, p[3] * gsf]

        center = [(p[0] + origin[0] * grid_scale) / grid_ratio, (origin[1] * grid_scale - p[1]) / grid_ratio]

        arrow_points = [[int(round(center[0] - scaled_vec[0] / 2)), int(round(center[1] + scaled_vec[1] / 2))],
                        [int(round(center[0] + scaled_vec[0] / 2)), int(round(center[1] - scaled_vec[1] / 2))]]

        cv2.arrowedLine(canvas, arrow_points[0], arrow_points[1], [0, 0, 0], 1, line_type=cv2.LINE_AA, tipLength=0.5)

        scaled_vec = [p[2] * grid_scale, p[3] * grid_scale]
        center = [(p[0] + sim_width / 2) / grid_ratio, pattern_canvas.shape[0] / 2.0]

        arrow_points = [[int(round(center[0] - scaled_vec[0] / 2)), int(round(center[1] + scaled_vec[1] / 2))],
                        [int(round(center[0] + scaled_vec[0] / 2)), int(round(center[1] - scaled_vec[1] / 2))]]

        cv2.arrowedLine(pattern_canvas, arrow_points[0], arrow_points[1], [255, 255, 255], 1, line_type=cv2.LINE_AA,
                        tipLength=0.4)


def update_canvas():
    cv2.imshow("Rankine Sim", canvas)
    cv2.imshow("Pattern", pattern_canvas)
    cv2.imshow("sim_window", settings_canvas)


def set_Vt(val):
    global Vt
    Vt = max(1, val)
    run_sim()


def set_Vr(val):
    global Vr
    Vr = max(1, val)
    run_sim()


def set_Vs(val):
    global Vs
    Vs = max(1, val)
    run_sim()


def set_Vc(val):
    global Vc
    Vc = max(10, val)
    run_sim()


def set_Rmax(val):
    global Rmax
    Rmax = max(10, val)
    run_sim()


def set_grid_scale(val):
    global grid_scale
    grid_scale = max(5, val)
    run_sim()


def set_Phi(val):
    global Phi
    val *= 0.05
    val += 0.5
    Phi = max(0.5, val)
    print(Phi)
    run_sim()


def run_sim():
    global grid_width
    global grid_height
    global vec_grid
    global pattern
    global pattern_canvas
    global canvas
    global settings_canvas

    grid_width = int(sim_width / grid_scale) + 1
    grid_height = int(sim_height / grid_scale) + 1
    vec_grid = [[-1] * grid_width for i in range(grid_height)]
    #pattern = [-1] * grid_width

    origin = [grid_width // 2, grid_height // 2]  # // is integer division

    pattern_canvas = np.zeros((grid_scale * 2, width, 3), np.uint8)

    #while origin[1] > grid_height / -2.0:
    generate_field(origin)

    pattern, Xc = generate_pattern(grid_scale * grid_ratio, Vr, Vt, Vs, Vc, Rmax, Phi)
    #pattern = RankineFastLib.generatePattern(sim_width / 2, grid_scale * grid_ratio, Vt, Vr, Vs, Vcrit, Rmax)

    #print(w[0], w[1])

    #sim_tree_fall(origin)
    render_field_mag()

    cv2.circle(canvas, [int(width / 2), int(height / 2)], int(Rmax / grid_ratio), [255, 0, 255], 2)
    Vmax = solve_vmax(Vt, Vr, Vs)
    settings_canvas = np.zeros((100, 300, 3), np.uint8)
    cv2.putText(settings_canvas, "Vmax: " + str(round(Vmax)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2, cv2.LINE_AA)

    render_pattern(pattern, Xc, origin)

    update_canvas()


if __name__ == '__main__':

    cv2.namedWindow("sim_window")
    cv2.createTrackbar("Phi", "sim_window", 10, 20, set_Phi)
    cv2.createTrackbar("Vt", "sim_window", 36, 50, set_Vt)
    cv2.createTrackbar("Vr", "sim_window", 72, 100, set_Vr)
    cv2.createTrackbar("Vs", "sim_window", 23, 30, set_Vs)
    cv2.createTrackbar("Vc", "sim_window", 40, 65, set_Vc)
    cv2.createTrackbar("Rmax", "sim_window", 200, 500, set_Rmax)
    cv2.createTrackbar("Grid_Scale", "sim_window", 25, 100, set_grid_scale)

    run_sim()

    cv2.waitKey(0)




