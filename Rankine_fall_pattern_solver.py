import numpy as np
import cv2
import math
import RankineFastLib
import time
import random

#simulation parameters
Vt = 36
Vr = 72
Vs = 23
Rmax = 200
dt = 0.001
Vcrit = 40

width, height = 1001, 1001 #pixels
canvas = np.zeros((height, width, 3), np.uint8)

sim_width, sim_height = 2000, 2000
grid_ratio = sim_width / width
grid_scale = 25 #meters

pattern_canvas = np.zeros((grid_scale * 2, width, 1), np.uint8)

grid_width = 0
grid_height = 0

vec_grid = []

trees = []
fallen_trees = []

pattern = []


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

            scale_factor = r * (1 / Rmax) if r <= Rmax else (1 / r) * Rmax

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


def generate_pattern(origin):
    for i in range(grid_width):
        m = -9999999
        w = -9999999
        no_solution_count = 0
        x = (i - origin[0]) * grid_scale

        Rmax2 = Rmax * Rmax
        x2 = x * x
        Vt2 = Vt * Vt
        Vr2 = Vr * Vr
        Vs2 = Vs * Vs
        Vcrit2 = Vcrit * Vcrit

        minY = Rmax2 - x2

        # r <= Rmax
        a = (1 / Rmax2) * (Vr2 + Vt2)
        b = -2 * (1 / Rmax) * Vr * Vs
        c = a * x2 + 2 * (1 / Rmax) * Vt * Vs * x + Vs2 - Vcrit2

        d = b*b - 4*a*c

        if -0.001 <= c <= 0.001:
            m = -b / a
        elif d >= 0: #solution exists (not imaginary)
            sqrtd = math.sqrt(d)

            y1 = (-b + sqrtd) / (2 * a)
            y2 = (-b - sqrtd) / (2 * a)

            if y1 * y1 > minY:
                y1 = -99999999

            if y2 * y2 > minY:
                y2 = -99999999

            m = max(y1, y2)
        else:
            no_solution_count += 1

        #r > Rmax
        a = Rmax2 * (Vr2 + Vt2)
        b = 2 * Rmax * Vr * Vs
        c = 2 * Rmax * Vt * Vs
        d = Vs2 - Vcrit2

        q = b*b - 4*d*(a + x*(c + d * x))

        if -0.001 <= d <= 0.001:
            w = (a + c * x) / b

        elif q >= 0: #solution exists (not imaginary)
            sqrtq = math.sqrt(q)

            y1 = (b + sqrtq) / (2 * d)
            y2 = (b - sqrtq) / (2 * d)

            if y1 * y1 <= minY:
                y1 = -99999999

            if y2 * y2 <= minY:
                y2 = -99999999

            w = max(y1, y2)
        else:
            no_solution_count += 1

        if no_solution_count != 2:
            y = max(w, m)

            radial_distance_vec = [-x, -y]

            r = math.hypot(radial_distance_vec[0], radial_distance_vec[1])

            radial_unit_vec = [radial_distance_vec[0] / r, radial_distance_vec[1] / r]

            tangential_unit_vec = [radial_unit_vec[1], -radial_unit_vec[0]]

            scale_factor = r * (1 / Rmax) if m > w else (1 / r) * Rmax

            Vtan = Vt * scale_factor
            Vrad = Vr * scale_factor

            tangential_vec = [tangential_unit_vec[0] * Vtan, tangential_unit_vec[1] * Vtan]

            radial_vec = [radial_unit_vec[0] * Vrad, radial_unit_vec[1] * Vrad]

            velocity = [tangential_vec[0] + radial_vec[0], tangential_vec[1] + radial_vec[1] + Vs]

            mag = math.hypot(velocity[0], velocity[1])

            if Vcrit * 1.05 >= mag >= Vcrit * 0.95:
                pattern[i] = [x, y, velocity[0] / mag, velocity[1] / mag]


def sim_tree_fall(origin):

    for i in range(len(trees)-1, 0, -1):
        if trees[i][1] == origin[1] and trees[i][1] == origin[0]:
            continue

        radial_distance_vec = [(origin[0] - trees[i][0]) * grid_scale, (trees[i][1] - origin[1]) * grid_scale]

        r = math.hypot(radial_distance_vec[0], radial_distance_vec[1])

        radial_unit_vec = [radial_distance_vec[0] / r, radial_distance_vec[1] / r]

        tangential_unit_vec = [radial_unit_vec[1], -radial_unit_vec[0]]

        scale_factor = r * (1 / Rmax) if r <= Rmax else (1 / r) * Rmax

        Vtan = Vt * scale_factor
        Vrad = Vr * scale_factor

        tangential_vec = [tangential_unit_vec[0] * Vtan, tangential_unit_vec[1] * Vtan]

        radial_vec = [radial_unit_vec[0] * Vrad, radial_unit_vec[1] * Vrad]

        velocity = [tangential_vec[0] + radial_vec[0], tangential_vec[1] + radial_vec[1] + Vs]

        mag = math.hypot(velocity[0], velocity[1])

        if mag > Vcrit:
            fallen_trees.append([trees[i][0], trees[i][1], velocity[0] / mag, velocity[1] / mag])
            trees.pop(i)


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


def render_fallen_trees():
    for tree in fallen_trees:

        gsf = grid_scale / grid_ratio

        scaled_vec = [tree[2] * gsf * 0.75, tree[3] * gsf * 0.75]

        center = [tree[0] * gsf + gsf / 2, tree[1] * gsf + gsf / 2]

        arrow_points = [[int(round(center[0] - scaled_vec[0]/2)), int(round(center[1] + scaled_vec[1]/2))],
                        [int(round(center[0] + scaled_vec[0]/2)), int(round(center[1] - scaled_vec[1]/2))]]

        cv2.arrowedLine(canvas, arrow_points[0], arrow_points[1], [0, 0, 0], 1, line_type=cv2.LINE_AA, tipLength=0.5)


def render_pattern(origin):
    for p in pattern:

        if p == -1:
            continue

        gsf = grid_scale / grid_ratio
        scaled_vec = [p[2] * gsf * 0.75, p[3] * gsf * 0.75]

        center = [(p[0] + origin[0]*grid_scale) / grid_ratio, (origin[1]*grid_scale - p[1]) / grid_ratio]

        arrow_points = [[int(round(center[0] - scaled_vec[0] / 2)), int(round(center[1] + scaled_vec[1] / 2))],
                        [int(round(center[0] + scaled_vec[0] / 2)), int(round(center[1] - scaled_vec[1] / 2))]]

        cv2.arrowedLine(canvas, arrow_points[0], arrow_points[1], [0, 0, 0], 1, line_type=cv2.LINE_AA, tipLength=0.5)

        scaled_vec = [p[2] * grid_scale, p[3] * grid_scale]
        center = [(p[0] + origin[0] * grid_scale) / grid_ratio * 2, pattern_canvas.shape[0] / 2.0]

        arrow_points = [[int(round(center[0] - scaled_vec[0] / 2)), int(round(center[1] + scaled_vec[1] / 2))],
                        [int(round(center[0] + scaled_vec[0] / 2)), int(round(center[1] - scaled_vec[1] / 2))]]

        cv2.arrowedLine(pattern_canvas, arrow_points[0], arrow_points[1], [255], 1, line_type=cv2.LINE_AA, tipLength=0.4)


def render_pattern_fast(origin):
    for p in pattern:

        if p[1] == 0 and p[2] == 0 and p[3] == 0:
            continue

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

        cv2.arrowedLine(pattern_canvas, arrow_points[0], arrow_points[1], [255], 1, line_type=cv2.LINE_AA,
                        tipLength=0.4)


def update_canvas():
    cv2.imshow("Rankine Sim", canvas)
    cv2.imshow("Pattern", pattern_canvas)


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


def set_Vcrit(val):
    global Vcrit
    Vcrit = max(10, val)
    run_sim()


def set_Rmax(val):
    global Rmax
    Rmax = max(10, val)
    run_sim()


def set_grid_scale(val):
    global grid_scale
    grid_scale = max(5, val)
    run_sim()


def run_sim():
    global grid_width
    global grid_height
    global vec_grid
    global pattern
    global pattern_canvas
    global canvas

    grid_width = int(sim_width / grid_scale) + 1
    grid_height = int(sim_height / grid_scale) + 1
    vec_grid = [[-1] * grid_width for i in range(grid_height)]
    #pattern = [-1] * grid_width

    origin = [grid_width // 2, grid_height // 2]  # // is integer division

    pattern_canvas = np.zeros((grid_scale * 2, width, 1), np.uint8)

    #while origin[1] > grid_height / -2.0:
    generate_field(origin)

    #generate_pattern(origin)
    pattern = RankineFastLib.generatePattern(sim_width / 2, grid_scale * grid_ratio, Vt, Vr, Vs, Vcrit, Rmax)

    #sim_tree_fall(origin)
    render_field_mag()

    cv2.circle(canvas, [int(width / 2), int(height / 2)], int(Rmax / grid_ratio), [255, 0, 255], 2)
    Vmax = RankineFastLib.solveVmaxRankine(Vr, Vt, Vs, Vcrit, Rmax)

    print("Vmax:", round(Vmax[4]))
    #render_pattern(origin)
    render_pattern_fast(origin)
    #render_fallen_trees()
    #render_field_arrows()

    update_canvas()
    # origin[1] -= dt * Vs


if __name__ == '__main__':

    cv2.namedWindow("sim_window")
    cv2.createTrackbar("Vt", "sim_window", 36, 100, set_Vt)
    cv2.createTrackbar("Vr", "sim_window", 72, 100, set_Vr)
    cv2.createTrackbar("Vs", "sim_window", 23, 100, set_Vs)
    cv2.createTrackbar("Vcrit", "sim_window", 40, 100, set_Vcrit)
    cv2.createTrackbar("Rmax", "sim_window", 200, 500, set_Rmax)
    cv2.createTrackbar("Grid_Scale", "sim_window", 25, 100, set_grid_scale)

    #random trees
    # for i in range(3000):
    #     trees.append([random.random() * grid_width, random.random() * grid_height])

    #evenly spaced trees
    # n = 50
    # for i in range(n):
    #     for j in range(n):
    #         trees.append([j * grid_width/n, i*grid_height/n])

    run_sim()

    cv2.waitKey(0)




