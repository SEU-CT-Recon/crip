from crip.physics import Atten
import matplotlib.pyplot as plt

gd = Atten.fromBuiltIn('Gd')
mu = []
erange = range(10, 120)
for e in erange:
    mu.append(gd.mu[e] * 10)
plt.figure()
plt.plot(erange, mu)
plt.show()
