import RankineFastLib
import time

t = time.time_ns()
x = RankineFastLib.generatePattern(1000.0, 100.0, 36.0, 72.0, 23.0, 40.0, 200.0)
print(time.time_ns() - t)

