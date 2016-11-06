# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from npcm import npcm
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
plt.style.use('ggplot')

def _generateFig5():
    """
    Two close clusters, one big and the other small,
    :return:
    """

    x0=np.array([[1,2.5],[1,3],[1.5,2],[1.5,2.5],[1.5,3],[1.5,3.5],[2,2],[2,2.5],[2,3],[2,3.5],[2.5,2.5],[2.5,3]])
    y0=np.zeros(len(x0))
    x1=np.array([[3.5,2.75],[4.25,2],[4.25,2.75],[4.25,3.5],[5,2.75]])
    y1=np.zeros(len(x1))+1
    X=np.vstack((x0,x1))
    y=np.hstack((y0,y1))
    # Visualize the test data
    fig0, ax0 = plt.subplots()
    for label in range(2):
        ax0.plot(X[y == label][:,0], X[y == label][:,1], '.',
                 color=colors[label])
        ax0.set_xlim(0,5.5)
        ax0.set_ylim(0,5.5)
    # ax0.set_title('Test data: 200 points x3 clusters.')
    return X
if __name__=='__main__':
    X= _generateFig5()
    fig,ax=plt.subplots()
    clf= npcm(X, 2, 1.5, ax=ax, x_lim=(0, 5.5), y_lim=(0, 5.5), alpha_cut=0.2)
    # we should set "blit=False,repeat=False" or the program would fail. "init_func=clf.init_animation" plot the
    # background of each frame There is not much point to use blit=True, if most parts of your plot should be
    # refreshed. see http://stackoverflow.com/questions/14844223/python-matplotlib-blit-to-axes-or-sides-of-the
    # -figure
    # To begin with, if you're chaining the ticks, etc, there isn't much point in using blitting. Blitting is
    #  just a way to avoid re-drawing everything if only some things are changing. If everything is changing,
    # there's no point in using blitting. Just re-draw the plot.
    anim = animation.FuncAnimation(fig, clf, frames=clf.fit,
                                   init_func=clf.init_animation, interval=700, blit=False,repeat=False)

    plt.show()
    pass