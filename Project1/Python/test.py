import numpy as np
import itertools
import matplotlib.pyplot as plt
m = 5
n = 5
x = np.zeros(shape=(m, n))
plt.figure(figsize=(5.15, 5.15))
plt.clf()
plt.subplot(111)
marker = itertools.cycle(('o', 'v', '^', '<', '>', 's', '8', 'p'))

ax = plt.gca().set_prop_cycle(None) 
for i in range(1, n):
    x = np.dot(i, [1, 1.1, 1.2, 1.3])
    y = x ** 2
    #
    #for matplotlib before 1.5, use
    #color = next(ax._get_lines.color_cycle)
    #instead of next line (thanks to Jon Loveday for the update)
    #
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(x, y, linestyle='', markeredgecolor='none', marker=marker.next(), color=color)
    plt.plot(x, y, linestyle='-', color = color)
plt.ylabel(r'$y$', labelpad=6)
plt.xlabel(r'$x$', labelpad=6)
plt.show()