import numpy as np
import cv2
import math
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

sim_width, sim_height = 1500, 1500
grid_ratio = sim_width / width
grid_scale = 25 #meters

grid_width = 0
grid_height = 0

vec_grid = []

trees = []
fallen_trees = []


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


def update_canvas():
    cv2.imshow("Rankine Sim", canvas)


def set_dt(val):
    global dt
    dt = 0.001 + val * 0.001


def set_Vt(val):
    global Vt
    Vt = max(1, val)


def set_Vr(val):
    global Vr
    Vr = max(1, val)


def set_Vs(val):
    global Vs
    Vs = max(1, val)


def set_Vcrit(val):
    global Vcrit
    Vcrit = max(10, val)


def set_Rmax(val):
    global Rmax
    Rmax = max(10, val)


def set_grid_scale(val):
    global grid_scale
    grid_scale = max(5, val)


def run_sim(offset):
    global grid_width
    global grid_height

    origin = [grid_width // 2, grid_height * 1.5 - offset]  # // is integer division

    #while origin[1] > grid_height / -2.0:
    generate_field(origin)
    sim_tree_fall(origin)
    render_field_mag()
    render_fallen_trees()
    #render_field_arrows()
    update_canvas()
    # origin[1] -= dt * Vs


if __name__ == '__main__':

    cv2.namedWindow("sim_window")
    cv2.createTrackbar("dt", "sim_window", 4, 99,  set_dt)
    cv2.createTrackbar("Vt", "sim_window", 36, 100, set_Vt)
    cv2.createTrackbar("Vr", "sim_window", 72, 100, set_Vr)
    cv2.createTrackbar("Vs", "sim_window", 23, 100, set_Vs)
    cv2.createTrackbar("Vcrit", "sim_window", 40, 100, set_Vcrit)
    cv2.createTrackbar("Rmax", "sim_window", 200, 500, set_Rmax)
    cv2.createTrackbar("Grid_Scale", "sim_window", 25, 100, set_grid_scale)

    cv2.putText(canvas, "Press and Hold Enter to Simulate", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2, cv2.LINE_AA)
    update_canvas()

    count = 0

    grid_width = int(sim_width / grid_scale) + 1
    grid_height = int(sim_height / grid_scale) + 1

    vec_grid = [[-1] * grid_width for i in range(grid_height)]

    #random trees
    # for i in range(3000):
    #     trees.append([random.random() * grid_width, random.random() * grid_height])

    #evenly spaced trees
    n = 60
    for i in range(n):
        for j in range(n):
            trees.append([j * grid_width/n, i*grid_height/n])


    while True:
        while True:
            if cv2.waitKeyEx(0) == 13: #enter key
                run_sim(dt * Vs * count)
                count += 1
                break




