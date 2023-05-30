/* Fast Library for improving performance of python code with c++ in automated tree tagging software
*  @author Daniel Butt, NTP 2022
*  @date Aug 9, 2022
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <omp.h>


#define PI      3.1415927f
#define PI_2    1.5707963f

//displays a progressbar for a for loop
//i = current index of loop
//total = total iterations of loop
//void updateProgressBar(int i, int total) {
//    //assumes index starts at 1
//    ++i;
//    int barLength = 50;
//    float progress = (float)i / (float)total;
//    int pos = (int)(progress * (float)barLength);
//
//    std::cout << "|" << std::string(pos, (char)(219)) << std::string(barLength - pos, ' ') << "| " << i << "/" << total << "\r";
//}

//nope better in hardware
//inline float sqrt_fast(float x) {
//
//    int e;
//    float m = frexpf(x, &e);
//
//    m = fmaf(m, fmaf(m, fmaf(m, 0.13392708f, -0.4997798f), 1.1009883f), 0.2649515f);
//
//    const bool odd = e & 1;
//
//    e = e >> 1;
//
//    m = odd ? 1.4142135f * m : m;
//
//    return ldexpf(m, e);
//}


inline float atan_fast(float x) {
    const float a1 = 0.995354f;
    const float a3 = -0.288679f;
    const float a5 = 0.079331f;

    // Compute approximation using Horner's method 
    const float x_sq = x * x;
    return x * fmaf(x_sq, fmaf(x_sq, a5, a3), a1);
}


float atan2_fast(float y, float x) {

    // Ensure input is in range [-1, +1]
    const bool swap = fabs(x) < fabs(y);
    const float atan_input = (swap ? x : y) / (swap ? y : x); 

    // Approximate atan
    float res = atan_fast(atan_input); 

    // If swapped, adjust atan output
    res = swap ? copysignf(PI_2, atan_input) - res : res; 

    // Adjust the result depending on the input quadrant
    res = x < 0.0f ? copysignf(PI, y) + res : res; 

    return res;
}

float polar_fast(float y, float x) {

    // Ensure input is in [-1, +1]
    const bool swap = fabs(x) < fabs(y);
    const float atan_input = (swap ? x : y) / (swap ? y : x);

    // Approximate atan
    float res = atan_fast(atan_input);

    // If swapped, adjust atan output
    res = swap ? copysignf(PI_2, atan_input) - res : res;

    // Adjust the result depending on the input quadrant
    res = x < 0.0f ? copysignf(PI, y) + res : res;
    res = res < 0.0f ? 2.0f*PI + res : res;

    return res;
}


inline std::array<float, 5> calculateRankine(float x, float y, float Vt, float Vr, float Vs, float Rmax, bool inner) {

    const std::array<float, 2> radial_distance_vec = { -x, -y };

    const float r = hypot(radial_distance_vec[0], radial_distance_vec[1]);

    const std::array<float, 2> radial_unit_vec = { radial_distance_vec[0] / r, radial_distance_vec[1] / r };

    const std::array<float, 2> tangential_unit_vec = { radial_unit_vec[1], -radial_unit_vec[0] };

    const float scaleFactor = inner ? r / Rmax : Rmax / r;

    const float Vtan = Vt * scaleFactor;
    const float Vrad = Vr * scaleFactor;

    const std::array<float, 2> vel = { tangential_unit_vec[0] * Vtan + radial_unit_vec[0] * Vrad , tangential_unit_vec[1] * Vtan + radial_unit_vec[1] * Vrad + Vs };

    const float mag = hypot(vel[0], vel[1]);

    //ensure solution is valid (after accounting for floating point rounding error)
    return { x, y, vel[0] / mag, vel[1] / mag, mag };

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

    //r <= Rmax inner
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


    //r > Rmax outer
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

        const float r = hypot(radial_distance_vec[0], radial_distance_vec[1]);

        const std::array<float, 2> radial_unit_vec = { radial_distance_vec[0] / r, radial_distance_vec[1] / r };

        const std::array<float, 2> tangential_unit_vec = { radial_unit_vec[1], -radial_unit_vec[0] };

        const float scaleFactor = innerSolution > outerSolution ? r / Rmax : Rmax / r;

        const float Vtan = Vt * scaleFactor;
        const float Vrad = Vr * scaleFactor;

        const std::array<float, 2> vel = { tangential_unit_vec[0] * Vtan + radial_unit_vec[0] * Vrad , tangential_unit_vec[1] * Vtan + radial_unit_vec[1] * Vrad + Vs };

        const float mag = hypot(vel[0], vel[1]);

        /*const float angle = atan2f(vel[1], vel[0]);

        const float angleRectified = angle > 0 ? angle : 2 * PI + angle;*/

        //const float angle = polar_fast(vel[1], vel[0]);

        //ensure solution is valid (after accounting for floating point rounding error)
        if (Vc * 0.95 < mag && mag < Vc * 1.05) {
            return { x, y, vel[0] / mag, vel[1] / mag};
        }

    }

    //no valid solution
    return { x, 0.0f, 0.0f, 0.0f};

}


std::array<float, 4> solveRankineInner(float x, float Vt, float Vr, float Vs, float Vc, float Rmax) {
    float innerSolution = -1 * std::numeric_limits<float>::max();

    const float x2 = x * x;
    const float Vt2 = Vt * Vt;
    const float Vr2 = Vr * Vr;
    const float Vs2 = Vs * Vs;
    const float Vc2 = Vc * Vc;
    const float Rmax2 = Rmax * Rmax;

    const float minY = Rmax2 - x2;

    const float a = (1 / Rmax2) * (Vr2 + Vt2);

    const float s = 2 * (1 / Rmax) * Vs;

    const float b = -s * Vr;
    const float c = a * x2 + s * Vt * x + Vs2 - Vc2;

    const float d = b * b - 4 * a * c;

    if (-0.001f <= c && c <= 0.001f) {
        innerSolution = -b / a;
    }
    else if (d >= 0) { // solution exists (not imaginary)
        const float sqrtd = sqrt(d);

        const float y1 = (-b + sqrtd) / (2 * a);
        const float y2 = (-b - sqrtd) / (2 * a);

        if (y1 * y1 > minY) {
            if (y2 * y2 <= minY) {
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

    if (innerSolution != -1 * std::numeric_limits<float>::max()) {
        const std::array<float, 5> solution = calculateRankine(x, innerSolution, Vt, Vr, Vs, Rmax, true);

        //ensure solution is valid (after accounting for floating point rounding error)
        if (Vc * 0.99 < solution[4] && solution[4] < Vc * 1.01) {
            return { solution[0], solution[1], solution[2], solution[3] };
        }
    }

    //no valid solution
    return { x, 0.0f, 0.0f, 0.0f };

}


std::array<float, 4> solveRankineOuter(float x, float Vt, float Vr, float Vs, float Vc, float Rmax) {
    float outerSolution = -1 * std::numeric_limits<float>::max();

    const float x2 = x * x;
    const float Vt2 = Vt * Vt;
    const float Vr2 = Vr * Vr;
    const float Vs2 = Vs * Vs;
    const float Vc2 = Vc * Vc;
    const float Rmax2 = Rmax * Rmax;

    const float minY = Rmax2 - x2;

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
        const float sqrtq = sqrt(q);

        const float y1 = (b + sqrtq) / (2 * d);
        const float y2 = (b - sqrtq) / (2 * d);

        if (y1 * y1 <= minY) {
            if (y2 * y2 > minY) {
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

    if (outerSolution != -1 * std::numeric_limits<float>::max()) {
        const std::array<float, 5> solution = calculateRankine(x, outerSolution, Vt, Vr, Vs, Rmax, false);

        //ensure solution is valid (after accounting for floating point rounding error)
        if (Vc * 0.99 < solution[4] && solution[4] < Vc * 1.01) {
            return { solution[0], solution[1], solution[2], solution[3]};
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

    const float hypotrt = hypot(Vr, Vt);
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

    const float mag = hypot(vel[0], vel[1]);


    return {x, y, vel[0] / mag, vel[1] / mag, mag};
}


std::array<float, 2> solvePatternAsymptotes(float Vt, float Vr, float Vs, float Vc, float Rmax) {
    const float Rmax2 = Rmax * Rmax;

    const float a = Rmax2 * (Vr * Vr + Vt * Vt);
    const float s = 2 * Rmax * Vs;
    const float b = s * Vr;
    const float c = s * Vt;
    const float d = Vs * Vs - Vc * Vc;
    const float a2 = a * a;
    const float b2 = b * b;
    const float c2 = c * c;

    const float discriminant = b2 * (Rmax2 * (b2 + c2 - d * (2 * a + d * Rmax2)) - a2);

    if (-0.001f <= discriminant && discriminant <= 0.001f) {
        return { -c * (a + d * Rmax2) / (b2 + c2), -std::numeric_limits<float>::max() };
    }
    else if (discriminant > 0.0f) {
        const float sqrtDis = sqrt(discriminant);
        const float n = -c * (a + d * Rmax2);
        const float m = b2 + c2;

        return { (n + sqrtDis) / m,  (n - sqrtDis) / m };

    }
    else {
        return { -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };
    }

}


std::array<std::array<float, 4>, 2> solveConvergenceRankine(float Vt, float Vr, float Vs, float Vc, float Rmax) {

    std::array<float, 2> asymptotes = solvePatternAsymptotes(Vt, Vr, Vs, Vc, Rmax);

    //outer solutions only
    if (asymptotes[0] == -std::numeric_limits<float>::max()) {
        const float Vt2 = Vt * Vt;
        const float Vr2 = Vr * Vr;
        const float Vs2 = Vs * Vs;
        const float Vc2 = Vc * Vc;
        const float Rmax2 = Rmax * Rmax;
        
        const float l = Vt / Vr;
        const float l2 = l * l;
        const float l2_1 = l2 + 1;
        
        const float n = Vs2 - Vc2;

        const float a = Rmax2 * (Vr2 + Vt2);
        const float s = 2 * Rmax * Vs;
        const float b = s * Vr;
        const float c = s * Vt;
        
        const float g = c * l + b;
        
        if (-0.001 <= n && n <= 0.001) {
                    
            const float x = (-l * a) / g;

            return { solveRankineOuter(x, Vt, Vr, Vs, Vc, Rmax) , {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()} };
        }
        else {
            const float sqrtp = sqrt((l2 * g * g) / (n * n) - (4 * a * l2 * l2_1) / n);
            const float k = (l * g) / n;

            const float x1 = (sqrtp - k) / (2 * l2_1);
            const float x2 = -(sqrtp + k) / (2 * l2_1);

            const std::array<std::array<float, 4>, 2> solution1 = { solveRankineOuter(x1, Vt, Vr, Vs, Vc, Rmax) , {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()} };
            const std::array<std::array<float, 4>, 2> solution2 = { solveRankineOuter(x2, Vt, Vr, Vs, Vc, Rmax) , {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()} };

            return (0.99f <= fabs(solution1[0][3]) && fabs(solution1[0][3]) <= 1.01f) ? solution1 : solution2;

            //return { solveRankineOuter(x1, Vt, Vr, Vs, Vc, Rmax) , solveRankineOuter(x2, Vt, Vr, Vs, Vc, Rmax) };
        }

    }
    //asymptotes exist, both inner and outer solutions
    else {
        const float x = std::max(asymptotes[0], asymptotes[1]);

        return { solveRankineOuter(x + 0.001f, Vt, Vr, Vs, Vc, Rmax), solveRankineInner(x - 0.001f, Vt, Vr, Vs, Vc, Rmax)};
    }
}


std::array<float, 2> solvePatternWidth(float Vt, float Vr, float Vs, float Vc, float Rmax) {
    const float a = -Vs * Vt * Rmax;
    const float b = Vr * Vr + Vt * Vt;
    const float c = Vc * Rmax * sqrtf(b);
    const float d = Vs * Vs - Vc * Vc;

    float xi0 = (a + c) / b;
    float xi1 = (a - c) / b;
    float xo0 = (a + c) / d;
    float xo1 = (a - c) / d;

    const auto s0 = solveRankineOuter(xo0 * 0.999f, Vt, Vr, Vs, Vc, Rmax);
    const auto s1 = solveRankineOuter(xo1 * 0.999f, Vt, Vr, Vs, Vc, Rmax);

    xi0 = fabs(xi0) > Rmax ? copysignf(Rmax, xi0) : xi0;
    xi1 = fabs(xi1) > Rmax ? copysignf(Rmax, xi1) : xi1;
    xo0 = (s0[2] == 0.0f && s0[3] == 0.0f) ? 0.0f : xo0;
    xo1 = (s1[2] == 0.0f && s1[3] == 0.0f) ? 0.0f : xo1;

    //return { xi0, xi1, xo0, xo1 };
    return { std::max(xi0, std::max(std::max(xo0, xo1), xi1)), std::min(xi0, std::min(std::min(xo0, xo1), xi1)) };
}


std::vector<std::array<float, 4>> generatePattern(float width, float gridScale, float Vt, float Vr, float Vs, float Vc, float Rmax) {

    std::vector<std::array<float, 4>> pattern = std::vector<std::array<float, 4>>();

    for (int x = ceil(-width / gridScale) * gridScale; x <= ceil(width / gridScale) * gridScale; x += gridScale) {

        pattern.push_back(solveRankine(x, Vt, Vr, Vs, Vc, Rmax));
    }

    return pattern;

}


std::vector<std::array<float, 6>> matchPattern(std::vector<std::array<float, 2>> p, const float observedW, const float wAbove, const float wBelow) {
    /*const float observedW = 600.0f;
    const float wAbove = 360.0f;
    const float wBelow = 240.0f;*/
    std::array<float, 5> matchedPattern;
    float lowestError = 10000000.0f;

    std::vector<std::array<float, 6>> bestPatterns;
    bestPatterns.reserve(10000);

    float bestError = 10000000.0f;

    bestPatterns.push_back({ 10000000.0f , -1.0f, -1.0f, -1.0f, -1.0f, -1.0f });

    //float p[16][2] = { {-0.882947593f, -0.469471563f}, {-0.891006524f, -0.4539905f}, {-0.866025404f, -0.5f}, {-0.121869343f, -0.992546152f}, {-0.087155743f, -0.996194698f}, {0.838670568f, -0.544639035f}, {0.992546152f, 0.121869343f}, {0.809016994f, 0.587785252f}, {0.838670568f, 0.544639035f}, {0.819152044f, 0.573576436f}, {0.838670568f, 0.544639035f}, {0.951056516f, 0.309016994f}, {0.891006524f, 0.4539905f}, {0.838670568f, 0.544639035f}, {0.529919264f, 0.848048096f}, {0.64278761f, 0.766044443f} };


    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for schedule(dynamic, 3)
    for (int Vt = 20; Vt < 80; Vt ++) {
        for (int Vr = 20; Vr < 80; Vr += 1) {
            for (int Vs = 10; Vs < 30; Vs += 1) {
                for (int Vc = 35; Vc < 55; Vc += 1) {
                    for (int Rmax = 80; Rmax < 400; Rmax += 1) {

                        //std::vector<std::array<float, 4>> pattern = std::vector<std::array<float, 4>>();

                        //find x coordinate of convergence 
                        const float c = solveConvergenceRankine(Vt, Vr, Vs, Vc, Rmax)[0][0];
                        //calculate pattern width
                        const std::array<float, 2> widthPoints = solvePatternWidth(Vt, Vr, Vs, Vc, Rmax);

                        const float w = widthPoints[0] - widthPoints[1];
                        const float delta = 40.0f;

                        //if width is not in viable range skip it
                        if (w > observedW + 3.0f * delta || w < observedW - 3.0f * delta) {
                            continue;
                        }

                        const float above = widthPoints[0] - c;
                        const float below = c - widthPoints[1];
                        

                        //if width above/below convergence line is not in viable range
                        if (above < wAbove - 2.0f*delta || above > wAbove + 2.0f * delta || below < wBelow - 2.0f * delta || below > wBelow + 2.0f * delta) {
                            continue;
                        }

                        int j = 0;
                        float error = 0.0f;

                        //calculate error below convergence line
                        for (float i = -1.0f * wBelow; i < 0.0f; i += delta, j++) {
                            const std::array<float, 4> pattern = solveRankine(c + i, Vt, Vr, Vs, Vc, Rmax);

                            if (pattern[2] == 0.0f && pattern[3] == 0.0f) {
                                error += 4.0f;
                            }
                            else {
                                const float d = -p[j][1] * pattern[2] + p[j][0] * pattern[3] - 1;
                                error += d * d;
                            }
                        }

                        //calcuate error around convergence line
                        const std::array<float, 4> pattern1 = solveRankine(c + 0.1f, Vt, Vr, Vs, Vc, Rmax);
                        const std::array<float, 4> pattern2 = solveRankine(c - 0.1f, Vt, Vr, Vs, Vc, Rmax);
                        const float d1 = -p[j][1] * pattern1[2] + p[j][0] * pattern1[3] - 1;
                        const float d2 = -p[j][1] * pattern2[2] + p[j][0] * pattern2[3] - 1;
                        const float dMin = std::min(d1, d2);
                        error += dMin * dMin;
                        j++;

                        //calculate error above convergence line
                        for (float i = delta; i <= wAbove + 0.1f; i += delta, j++) {
                            const std::array<float, 4> pattern = solveRankine(c + i, Vt, Vr, Vs, Vc, Rmax);

                            if (pattern[2] == 0.0f && pattern[3] == 0.0f) {
                                error += 4.0f;
                            }
                            else {
                                const float d = -p[j][1] * pattern[2] + p[j][0] * pattern[3] - 1;
                                error += d * d;
                            }
                        }


                        //calculate pattern
                        /*for (float i = -240.0f; i <= 360.1f; i += delta) {
                            pattern.push_back(solveRankine(c + i, Vt, Vr, Vs, Vc, Rmax));
                        }*/

                        /*const float we = (1 - w / observedW);
                        const float werror = 16 * we * we;*/

                        //compute error 
                        /*for (int i = 0; i < pattern.size(); i++) {
                            if (pattern[i][2] == 0.0f && pattern[i][3] == 0.0f) {
                                error += 4.0f;
                            }
                            else {
                                const float d = -p[i][1] * pattern[i][2] + p[i][0] * pattern[i][3] - 1;
                                error += d * d;
                            }
                        }*/

                        //adjust error based on number of vectors in patttern
                        error /= 16.0f;

                        //update list of best patterns (within 5% of best pattern so far)
                        #pragma omp critical
                        {
                            if (error < 1.05f * bestError) {
                                std::array<float, 6> item = { error, (float)Vt, (float)Vr, (float)Vs, (float)Vc, (float)Rmax };

                                //Red and Black tree or some other tree data structure is probably better than a vector here
                                bestPatterns.insert(std::upper_bound(bestPatterns.begin(), bestPatterns.end(), item), item);

                                if (error < bestError) {
                                    bestError = error;
                                    std::array<float, 6> lastItem = { 1.05f * bestError, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

                                    auto ub = std::upper_bound(bestPatterns.begin(), bestPatterns.end(), lastItem);

                                    bestPatterns.erase(ub, bestPatterns.end());

                                    /*while (bestPatterns[bestPatterns.size() - 1][0] > 1.05 * bestError) {
                                        bestPatterns.pop_back();
                                    }*/

                                }
                            }
                        }

                    }
                }
            }
        }
    }

    return bestPatterns;

}
    

//std::vector<std::vector<std::array<float, 5>>> generatePattern(float width, float gridScale, float Vt, float Vr, float Vs, float Vc, float Rmax) {
//
//    std::vector<std::vector<std::array<float, 5>>> results = std::vector<std::vector<std::array<float, 5>>>();
//    results.reserve(10);
//
//    
//    for (int i = 0; i < 10000000; i++) {
//        //updateProgressBar(i, 1000000);
//
//        Vt += 0.000001f;
//        Vr += 0.000001f;
//        Vs -= 0.000001f;
//        Vc -= 0.000001f;
//        Rmax += 0.000001f;
//
//        std::vector<std::array<float, 5>> pattern = std::vector<std::array<float, 5>>();
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
    handle.doc() = "Fast Library for improving performance of python code with compiler optimized c++";
    handle.def("generatePattern", &generatePattern);
    handle.def("solveVmaxRankine", &solveVmaxRankine);
    handle.def("solveConvergenceRankine", &solveConvergenceRankine);
    handle.def("solvePatternAsymptotes", &solvePatternAsymptotes);
    handle.def("solvePatternWidth", &solvePatternWidth);
    handle.def("matchPattern", &matchPattern);
}