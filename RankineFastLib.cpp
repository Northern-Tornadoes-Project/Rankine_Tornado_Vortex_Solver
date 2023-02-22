/* Fast Library for improving performance of python code with c++ in automated tree tagging software
*  @author Daniel Butt, NTP 2022
*  @date Aug 9, 2022
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_set>


#define PI 3.141592653f

//displays a progressbar for a for loop
//i = current index of loop
//total = total iterations of loop
void updateProgressBar(int i, int total) {
    //assumes index starts at 1
    ++i;
    int barLength = 50;
    float progress = (float)i / (float)total;
    int pos = (int)(progress * (float)barLength);

    std::cout << "|" << std::string(pos, (char)(219)) << std::string(barLength - pos, ' ') << "| " << i << "/" << total << "\r";
}


std::array<float, 4> solveRankine(float x, float Vt, float Vr, float Vs, float Vc, float Rmax) {
    float innerSolution = -1 * std::numeric_limits<float>::max();
    float outerSolution = -1 * std::numeric_limits<float>::max();

    int noSolutionCount = 0;

    const float x2 = x * x;
    const float Vt2 = Vt * Vt;
    const float Vr2 = Vr * Vr;
    const float Vs2 = Vs * Vs;
    const float Vc2 = Vc * Vc;
    const float Rmax2 = Rmax * Rmax;

    const float minY = Rmax2 - x2;

    //r <= Rmax
    {
        const float a = (1 / Rmax2) * (Vr2 + Vt2);

        const float s = 2 * (1 / Rmax) * Vs;

        const float b = -s * Vr;
        const float c = a * x2 + s * Vt * x + Vs2 - Vc2;

        const float d = b * b - 4 * a * c;

        if (-0.001f <= c && c <= 0.001f) {
            innerSolution = -b / a;
        }
        else if (d >= 0) { // solution exists (not imaginary)
            const float sqrtd = sqrtf(d);

            const float y1 = (-b + sqrtd) / (2 * a);
            const float y2 = (-b - sqrtd) / (2 * a);

            if (y1 * y1 > minY) {
                if (y2 * y2 > minY) {
                    noSolutionCount++;
                }
                else
                {
                    innerSolution = y2;
                }
            }
            else if (y2 * y2 > minY) {
                innerSolution = y1;
            }
            else {
                innerSolution = std::max(y1, y2);
            }

        }
        else {
            noSolutionCount++;
        }
    }


    //r > Rmax
    {
        const float a = Rmax2 * (Vr2 + Vt2);
        const float s = 2 * Rmax * Vs;
        const float b = s * Vr;
        const float c = s * Vt;
        const float d = Vs2 - Vc2;

        const float q = b * b - 4 * d * (a + x * (c + d * x));

        if (-0.001 <= d && d <= 0.001) {
            outerSolution = (a + c * x) / b;
        }
        else if (q >= 0) { //solution exists (not imaginary)
            const float sqrtq = sqrtf(q);

            const float y1 = (b + sqrtq) / (2 * d);
            const float y2 = (b - sqrtq) / (2 * d);

            if (y1 * y1 <= minY) {
                if (y2 * y2 <= minY) {
                    noSolutionCount++;
                }
                else
                {
                    outerSolution = y2;
                }
            }
            else if (y2 * y2 <= minY) {
                outerSolution = y1;
            }
            else {
                outerSolution = std::max(y1, y2);
            }
        }
        else {
            noSolutionCount++;
        }
    }

    //if solution exists create solution vector
    if (noSolutionCount != 2) {
        const float y = std::max(innerSolution, outerSolution);

        const std::array<float, 2> radial_distance_vec = { -x, -y };

        const float r = hypotf(radial_distance_vec[0], radial_distance_vec[1]);

        const std::array<float, 2> radial_unit_vec = { radial_distance_vec[0] / r, radial_distance_vec[1] / r };

        const std::array<float, 2> tangential_unit_vec = { radial_unit_vec[1], -radial_unit_vec[0] };

        const float scaleFactor = innerSolution > outerSolution ? r * (1 / Rmax) : (1 / r) * Rmax;

        const float Vtan = Vt * scaleFactor;
        const float Vrad = Vr * scaleFactor;

        const std::array<float, 2> vel = { tangential_unit_vec[0] * Vtan + radial_unit_vec[0] * Vrad , tangential_unit_vec[1] * Vtan + radial_unit_vec[1] * Vrad + Vs };

        const float mag = hypotf(vel[0], vel[1]);

        /*const float angle = atan2f(vel[1], vel[0]);

        const float angleRectified = angle > 0 ? angle : 2 * PI + angle;*/

        //ensure solution is valid (after accounting for floating point rounding error)
        if (Vc * 0.95 < mag && mag < Vc * 1.05) {
            return { x, y, vel[0] / mag, vel[1] / mag};
        }

    }

    //no valid solution
    return { x, 0.0f, 0.0f, 0.0f };

}


std::array<float, 5> solveVmaxRankine(float Vt, float Vr, float Vs, float Vc, float Rmax) {

    /*const float Rmax_1 = 1 / Rmax;

    const float a = Rmax_1 * Rmax_1 * (Vr * Vr + Vt * Vt);

    const float b = Rmax_1 * Vt * Vs;

    const float c = Rmax_1 * Vr * Vs;

    const float minX = -b / a;
    const float minY = c / a;

    const float minDist = hypotf(minX, minY);

    const float x = minX / minDist * -Rmax;
    const float y = minY / minDist * -Rmax;*/

    const float hypotrt = hypotf(Vr, Vt);
    const float s = Rmax / hypotrt;

    const float x = Vt * s;
    const float y = -Vr * s;

    const std::array<float, 2> radial_distance_vec = { -x, -y };

    const float r = Rmax;

    const std::array<float, 2> radial_unit_vec = { radial_distance_vec[0] / r, radial_distance_vec[1] / r };

    const std::array<float, 2> tangential_unit_vec = { radial_unit_vec[1], -radial_unit_vec[0] };

    const float scaleFactor = r * (1 / Rmax);

    const float Vtan = Vt * scaleFactor;
    const float Vrad = Vr * scaleFactor;

    const std::array<float, 2> vel = { tangential_unit_vec[0] * Vtan + radial_unit_vec[0] * Vrad , tangential_unit_vec[1] * Vtan + radial_unit_vec[1] * Vrad + Vs };

    const float mag = hypotf(vel[0], vel[1]);


    return {x, y, vel[0] / mag, vel[1] / mag, mag};
}


std::vector<std::array<float, 4>> generatePattern(float width, float gridScale, float Vt, float Vr, float Vs, float Vc, float Rmax) {

    std::vector<std::array<float, 4>> pattern = std::vector<std::array<float, 4>>();

    for (int x = ceil(-width / gridScale) * gridScale; x <= ceil(width / gridScale) * gridScale; x += gridScale) {

        pattern.push_back(solveRankine(x, Vt, Vr, Vs, Vc, Rmax));
    }

    return pattern;

}


//std::vector<std::vector<std::array<float, 4>>> generatePattern(float width, float gridScale, float Vt, float Vr, float Vs, float Vc, float Rmax) {
//
//    std::vector<std::vector<std::array<float, 4>>> results = std::vector<std::vector<std::array<float, 4>>>();
//    results.reserve(10);
//
//    
//    for (int i = 0; i < 100000000; i++) {
//        //updateProgressBar(i, 1000000);
//
//        Vt += 0.000001f;
//        Vr += 0.000001f;
//        Vs -= 0.000001f;
//        Vc -= 0.000001f;
//        Rmax += 0.000001f;
//
//        std::vector<std::array<float, 4>> pattern = std::vector<std::array<float, 4>>();
//
//        for (int x = ceil(-width / gridScale) * gridScale; x <= ceil(width / gridScale) * gridScale; x += gridScale) {
//
//            pattern.push_back(solveRankine(x, Vt, Vr, Vs, Vc, Rmax));
//        }
//
//        if (i > 9) {
//            if (results[i % 10][0][4] > pattern[0][4]) {
//                results[i % 10] = pattern;
//            }
//        }
//        else {
//            results.push_back(pattern);
//        }
//    }
//
//    return results;
//}


//pybind 11 boilerplate for compiling to python binary
PYBIND11_MODULE(RankineFastLib, handle) {
    handle.doc() = "Fast Library for improving performance of python code with c++ in automated tree tagging software";
    handle.def("generatePattern", &generatePattern);
    handle.def("solveVmaxRankine", &solveVmaxRankine);
    
}