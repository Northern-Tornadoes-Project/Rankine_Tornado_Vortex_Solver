import RankineFastLib
import time
import math
import numpy

import psutil
import os


p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)
#

# x = RankineFastLib.generatePattern(1000.0, 40.0, 36.0, 72.0, 23.0, 40.0, 200.0)
# print(time.time() - t)
# print(len(x))
#
# y = RankineFastLib.solveVmaxRankine(36.0, 72.0, 23.0, 40.0, 200.0)
#
# print(numpy.float32(y[3]))

#print(RankineFastLib.solveConvergenceRankine(36.0, 72.0, 23.0, 40.0, 200.0))
#print(RankineFastLib.solveConvergenceRankine(80.0, 65.0, 23.0, 45.0, 178.0))
# print(RankineFastLib.solveConvergenceRankine(21.0, 41.0, 11.0, 40.0, 200.0))

# x = RankineFastLib.solveConvergenceRankine(21.0, 68.0, 11.0, 40.0, 200.0)
#
# for v in x:
#     if 0.99 <= abs(v[3]) <= 1.01:
#         print(v)
#print(RankineFastLib.solvePatternAsymptotes(23.0, 54.0, 17.0, 54.0, 300.0))
#print(RankineFastLib.solvePatternWidth(36.0, 72.0, 23.0, 35.0, 200.0))

p = [[-0.882947593, -0.469471563], [-0.891006524, -0.4539905], [-0.866025404, -0.5], [-0.121869343, -0.992546152], [-0.087155743, -0.996194698], [0.838670568, -0.544639035], [0.992546152, 0.121869343], [0.809016994, 0.587785252], [0.838670568, 0.544639035], [0.819152044, 0.573576436], [0.838670568, 0.544639035], [0.951056516, 0.309016994], [0.891006524, 0.4539905], [0.838670568, 0.544639035], [0.529919264, 0.848048096], [0.64278761, 0.766044443]]


t = time.time()
pattern = RankineFastLib.matchPattern(p, 600.0, 360.0, 240.0)
print(len(pattern))
print(pattern)
print(time.time() - t)
# for p in pattern:
#     if p[2] != 0 and p[3] != 0:
#         print(math.degrees(math.atan2(p[3], p[2])))
#     else:
#         print("x")

