import numpy as np
import cv2
import math
import time

#simulation parameters
Vt = 36
Vr = 72
Vs = 23
Rmax = 200
Phi = 1.0

width, height = 1001, 1001 #pixels
canvas = np.zeros((height, width, 3), np.uint8)

sim_width, sim_height = 1500, 1500
grid_ratio = sim_width / width
grid_scale = 25 #meters

grid_width = 0
grid_height = 0

vec_grid = []


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

            scale_factor = (r * (1 / Rmax))**Phi if r <= Rmax else ((1 / r) * Rmax)**Phi

            Vtan = Vt * scale_factor
            Vrad = Vr * scale_factor

            tangential_vec = [tangential_unit_vec[0] * Vtan, tangential_unit_vec[1] * Vtan]

            radial_vec = [radial_unit_vec[0] * Vrad, radial_unit_vec[1] * Vrad]

            vec_grid[i][j] = [tangential_vec[0] + radial_vec[0], tangential_vec[1] + radial_vec[1] + Vs]


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


def render_field(origin):

    #color magnitudes
    for i in range(grid_height):
        for j in range(grid_width):
            mag = math.hypot(vec_grid[i][j][0], vec_grid[i][j][1])

            gsf = grid_scale/grid_ratio
            cv2.rectangle(canvas, (int(j*gsf), int(i*gsf)), (int(j*gsf + gsf), int(i*gsf + gsf)), mag_to_color(mag), -1)

    #vector arrows
    for i in range(grid_height):
        for j in range(grid_width):
            if i == origin[1] and j == origin[0]:
                vec_grid[i][j] = [0, 0]
                continue
            mag = math.hypot(vec_grid[i][j][0], vec_grid[i][j][1])

            if mag == 0:
                continue

            gsf = grid_scale / grid_ratio

            scaled_vec = [vec_grid[i][j][0] * gsf * 0.75 / mag, vec_grid[i][j][1] * gsf * 0.75 / mag]


            center = [j*gsf + gsf/2, i*gsf + gsf/2]
            arrow_points = [[int(round(center[0] - scaled_vec[0]/2)), int(round(center[1] + scaled_vec[1]/2))],
                            [int(round(center[0] + scaled_vec[0]/2)), int(round(center[1] - scaled_vec[1]/2))]]

            cv2.arrowedLine(canvas, arrow_points[0], arrow_points[1], [0, 0, 0], 1, line_type=cv2.LINE_AA, tipLength=0.5)


def display_field():
    cv2.imshow("Rankine Sim", canvas)


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
    run_sim()


def run_sim():

    global grid_width
    global grid_height

    grid_width = int(sim_width / grid_scale) + 1
    grid_height = int(sim_height / grid_scale) + 1

    global vec_grid
    vec_grid = [[-1] * grid_width for i in range(grid_height)]

    origin = [grid_width // 2, grid_height // 2]  # // is integer division

    generate_field(origin)
    render_field(origin)
    display_field()

    t = time.time_ns()
    points = polarRankine(Vr, Vt, Vs, 50.0, Rmax, Phi)
    print((time.time_ns() - t) * 1e-6)

    print(points)



def polarRankine(a, t, s, c, R, p):
    n = 100
    cartesian = []

    sqrt2 = math.sqrt(2)
    a2 = a * a
    t2 = t * t
    s2 = s * s
    c2 = c * c
    a2t2 = a2 + t2
    a2t2_1 = 1 / a2t2
    k1 = 2**(-1/p) * R
    k2 = -2*s*t
    k3 = 2*s*a
    k4 = (2*c2-s2) * a2t2
    k5 = -s2 * (a2 - t2)
    k6 = -2*a*s2*t

    xmin = 1000000
    xmax = -1000000
    dxmax = 0

    for i in range(n):
        sin = math.sin((i*2*math.pi)/float(n))
        cos = math.cos((i*2*math.pi)/float(n))
        sin2 = 2*sin*cos
        cos2 = cos * cos - sin * sin

        r = k1 * ((k2*cos + k3*sin + sqrt2 * math.sqrt(k4 + k5*cos2 + k6*sin2)) * a2t2_1)**(1/p)

        x = r*cos
        y = r*sin

        xmin = min(x, xmin)
        xmax = max(x, xmax)

        if i != 0:
            dxmax = max(abs(x - cartesian[len(cartesian) - 1][0]), dxmax)

        cartesian.append([x, y])

    return cartesian, xmin, xmax, dxmax


def findCircle(x1, y1, x2, y2, x3, y3):
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2)

    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2)

    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    f = ((sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) /
         (2 * (y31 * x12 - y21 * x13)))

    g = ((sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) /
         (2 * (x31 * y12 - x21 * y13)))

    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1)

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c

    # r is the radius
    r = round(math.sqrt(sqr_of_r), 5)

    print("Centre = (", h, ", ", k, ")")
    print("Radius = ", r)


if __name__ == '__main__':

    cv2.namedWindow("sim_window")
    cv2.createTrackbar("Vt", "sim_window", 36, 100, set_Vt)
    cv2.createTrackbar("Vr", "sim_window", 72, 100, set_Vr)
    cv2.createTrackbar("Vs", "sim_window", 23, 100, set_Vs)
    cv2.createTrackbar("Rmax", "sim_window", 200, 500, set_Rmax)
    cv2.createTrackbar("Phi", "sim_window", 10, 10, set_Phi)
    cv2.createTrackbar("Grid_Scale", "sim_window", 25, 100, set_grid_scale)

    findCircle(217.484485433, 0, 0, 305.489410109, -247.116224354, 0)

    run_sim()
    cv2.waitKey(0)


