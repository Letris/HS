import math

s = 7
t = 0.3

v1 = 1 * 0.9849
v2 = -0.2 * 0
v3 = 0.4 * 0.6500
v4 = 0.3 * 0.8590
v5 = -0.6 * 0.7524

part1 = 1 / (1 + math.exp(-s * (v1 + v2 - t)))
part2 = 1 / (1 + math.exp(s * t))
part3 = 1 + math.exp(-s * t)
aggimpact = (part1 - part2) * part3
print(part1, part2, part3)
print(aggimpact)