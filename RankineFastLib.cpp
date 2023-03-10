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


inline std::array<float, 5> calculateRankine(float x, float y, float Vt, float Vr, float Vs, float Rmax, bool inner) {

    const std::array<float, 2> radial_distance_vec = { -x, -y };

    const float r = hypotf(radial_distance_vec[0], radial_distance_vec[1]);

    const std::array<float, 2> radial_unit_vec = { radial_distance_vec[0] / r, radial_distance_vec[1] / r };

    const std::array<float, 2> tangential_unit_vec = { radial_unit_vec[1], -radial_unit_vec[0] };

    const float scaleFactor = inner ? r / Rmax : Rmax / r;

    const float Vtan = Vt * scaleFactor;
    const float Vrad = Vr * scaleFactor;

    const std::array<float, 2> vel = { tangential_unit_vec[0] * Vtan + radial_unit_vec[0] * Vrad , tangential_unit_vec[1] * Vtan + radial_unit_vec[1] * Vrad + Vs };

    const float mag = hypotf(vel[0], vel[1]);

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

        const float r = hypotf(radial_distance_vec[0], radial_distance_vec[1]);

        const std::array<float, 2> radial_unit_vec = { radial_distance_vec[0] / r, radial_distance_vec[1] / r };

        const std::array<float, 2> tangential_unit_vec = { radial_unit_vec[1], -radial_unit_vec[0] };

        const float scaleFactor = innerSolution > outerSolution ? r / Rmax : Rmax / r;

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
        const float sqrtd = sqrtf(d);

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
        const float sqrtq = sqrtf(q);

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


//std::vector<std::array<float, 4>> solveConvergenceRankine(float Vt, float Vr, float Vs, float Vc, float Rmax) {
//    std::array<std::array<float, 2>, 2> innerSolutions;
//    innerSolutions[0] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
//    innerSolutions[1] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
//
//    std::array<std::array<float, 2>, 2> outerSolutions;
//    outerSolutions[0] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
//    outerSolutions[1] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
//
//    const float Vt2 = Vt * Vt;
//    const float Vr2 = Vr * Vr;
//    const float Vs2 = Vs * Vs;
//    const float Vc2 = Vc * Vc;
//    const float Rmax2 = Rmax * Rmax;
//
//    const float l = Vt / Vr;
//    const float l2 = l * l;
//    const float l2_1 = l2 + 1;
//
//    const float n = Vs2 - Vc2;
//
//    std::vector<std::array<float, 4>> convVecs = std::vector<std::array<float, 4>>();
//
//
//    //r <= Rmax, inner solutions
//    {
//        const float a = (1 / Rmax2) * (Vr2 + Vt2);
//        const float s = 2 * (1 / Rmax) * Vs;
//        const float b = -s * Vr;
//        const float u = s * Vt;
//
//        const float g = l * u - b;
//
//        const float sqrtp = sqrtf((l2 * g * g) / (a * a) - (4 * n * l2 * l2_1) / a);
//        const float k = (l * g) / a;
//
//        //check validity
//        for (int i = -1; i <= 1; i += 2) {
//            const float x = (i*sqrtp - k) / (2 * l2_1);
//            std::cout << "inner: " << i << ", " << x << std::endl;
//            const std::array<float, 4> solution = solveRankine(x, Vt, Vr, Vs, Vc, Rmax);
//
//            if (solution[3] != 0.0f && solution[1] * solution[1] <= Rmax2 - solution[0] * solution[0]) {
//                convVecs.push_back(solution);
//            }
//
//            //const float x2 = x * x;
//            //const float minY = Rmax2 - x2;
//
//            //const float c = a * x2 + s * Vt * x + Vs2 - Vc2;
//            //const float d = b * b - 4 * a * c;
//
//            //if (-0.001f <= c && c <= 0.001f) {
//            //    innerSolutions[(i+1)/2][1] = -b / a;
//            //    innerSolutions[(i + 1) / 2][0] = x;
//            //}
//            //else if (d >= 0) { // solution exists (not imaginary)
//            //    const float sqrtd = sqrtf(d);
//
//            //    const float y1 = (-b + sqrtd) / (2 * a);
//            //    const float y2 = (-b - sqrtd) / (2 * a);
//
//            //    if (y1 * y1 > minY) {
//            //        if (y2 * y2 <= minY) {
//            //            innerSolutions[(i + 1) / 2][1] = y2;
//            //            innerSolutions[(i + 1) / 2][0] = x;
//            //        }
//            //    }
//            //    else if (y2 * y2 > minY) {
//            //        innerSolutions[(i + 1) / 2][1] = y1;
//            //        innerSolutions[(i + 1) / 2][0] = x;
//            //    }
//            //    else {
//            //        innerSolutions[(i + 1) / 2][1] = std::max(y1, y2);
//            //        innerSolutions[(i + 1) / 2][0] = x;
//            //    }
//
//            //}
//        }
//    }
//
//    //r > Rmax, outer solutions
//    {
//        const float a = Rmax2 * (Vr2 + Vt2);
//        const float s = 2 * Rmax * Vs;
//        const float b = s * Vr;
//        const float c = s * Vt;
//
//        const float g = c * l + b;
//
//        if (-0.001 <= n && n <= 0.001) {
//            
//            const float x = (-l * a) / g;
//            std::cout << "outer n=0: " << x << std::endl;
//            const std::array<float, 4> solution = solveRankine(x, Vt, Vr, Vs, Vc, Rmax);
//            if (solution[3] != 0.0f && solution[1] * solution[1] > Rmax2 - solution[0] * solution[0]) {
//                convVecs.push_back(solution);
//            }
//            /*outerSolutions[0][1] = (a + c * x) / b;
//            outerSolutions[0][0] = x;*/
//        }
//        else {
//            const float sqrtp = sqrtf((l2 * g * g) / (n * n) - (4 * a * l2 * l2_1) / n);
//            const float k = (l * g) / n;
//
//            //check validity
//            for (int i = -1; i <= 1; i += 2) {
//                const float x = (i * sqrtp - k) / (2 * l2_1);
//                std::cout << "outer: " << i << ", " << x << std::endl;
//                const std::array<float, 4> solution = solveRankine(x, Vt, Vr, Vs, Vc, Rmax);
//                if (solution[3] != 0.0f && solution[1] * solution[1] > Rmax2 - solution[0] * solution[0]) {
//                    convVecs.push_back(solution);
//                }
//                //const float x2 = x * x;
//                //const float minY = Rmax2 - x2;
//
//                //const float q = b * b - 4 * n * (a + x * (c + n * x));
//
//                //if (q >= 0) { //solution exists (not imaginary)
//                //    const float sqrtq = sqrtf(q);
//
//                //    const float y1 = (b + sqrtq) / (2 * n);
//                //    const float y2 = (b - sqrtq) / (2 * n);
//
//                //    if (y1 * y1 <= minY) {
//                //        if (y2 * y2 > minY) {
//                //            outerSolutions[(i + 1) / 2][1] = y2;
//                //            outerSolutions[(i + 1) / 2][0] = x;
//                //        }
//                //    }
//                //    else if (y2 * y2 <= minY) {
//                //        outerSolutions[(i + 1) / 2][1] = y1;
//                //        outerSolutions[(i + 1) / 2][0] = x;
//                //    }
//                //    else {
//                //        outerSolutions[(i + 1) / 2][1] = std::max(y1, y2);
//                //        outerSolutions[(i + 1) / 2][0] = x;
//                //    }
//                //}
//
//            }
//        }
//        
//    }
//
//    /*for (int i = 0; i < 2; i++) {
//        if (innerSolutions[i][0] != std::numeric_limits<float>::max()) {
//            const std::array<float, 2> radial_distance_vec = { -innerSolutions[i][0], -innerSolutions[i][1] };
//
//            const float r = hypotf(radial_distance_vec[0], radial_distance_vec[1]);
//
//            const std::array<float, 2> radial_unit_vec = { radial_distance_vec[0] / r, radial_distance_vec[1] / r };
//
//            const std::array<float, 2> tangential_unit_vec = { radial_unit_vec[1], -radial_unit_vec[0] };
//
//            const float scaleFactor = r * (1 / Rmax);
//
//            const float Vtan = Vt * scaleFactor;
//            const float Vrad = Vr * scaleFactor;
//
//            const std::array<float, 2> vel = { tangential_unit_vec[0] * Vtan + radial_unit_vec[0] * Vrad , tangential_unit_vec[1] * Vtan + radial_unit_vec[1] * Vrad + Vs };
//
//            const float mag = hypotf(vel[0], vel[1]);
//
//            convVecs.push_back( { innerSolutions[i][0], innerSolutions[i][1], vel[0] / mag, vel[1] / mag, mag });
//            
//        }
//    }
//
//    for (int i = 0; i < 2; i++) {
//        if (outerSolutions[i][0] != std::numeric_limits<float>::max()) {
//
//            const std::array<float, 2> radial_distance_vec = { -outerSolutions[i][0], -outerSolutions[i][1] };
//
//            const float r = hypotf(radial_distance_vec[0], radial_distance_vec[1]);
//
//            const std::array<float, 2> radial_unit_vec = { radial_distance_vec[0] / r, radial_distance_vec[1] / r };
//
//            const std::array<float, 2> tangential_unit_vec = { radial_unit_vec[1], -radial_unit_vec[0] };
//
//            const float scaleFactor = (1 / r) * Rmax;
//
//            const float Vtan = Vt * scaleFactor;
//            const float Vrad = Vr * scaleFactor;
//
//            const std::array<float, 2> vel = { tangential_unit_vec[0] * Vtan + radial_unit_vec[0] * Vrad , tangential_unit_vec[1] * Vtan + radial_unit_vec[1] * Vrad + Vs };
//
//            const float mag = hypotf(vel[0], vel[1]);
//
//            convVecs.push_back({ outerSolutions[i][0], outerSolutions[i][1], vel[0] / mag, vel[1] / mag, mag });
//        }
//    }*/
//
//    return convVecs;
//}


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
        const float sqrtDis = sqrtf(discriminant);
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
            const float sqrtp = sqrtf((l2 * g * g) / (n * n) - (4 * a * l2 * l2_1) / n);
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
    handle.doc() = "Fast Library for improving performance of python code with compiler optimized c++";
    handle.def("generatePattern", &generatePattern);
    handle.def("solveVmaxRankine", &solveVmaxRankine);
    handle.def("solveConvergenceRankine", &solveConvergenceRankine);
    handle.def("solvePatternAsymptotes", &solvePatternAsymptotes);
}