# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from npcm import npcm
from sklearn.datasets import make_blobs
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import os

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
plt.style.use('classic')


def _generateFig6():
    """
    Two close clusters, one big and the other small,
    :return:
    """

    x0, y0 = make_blobs(n_samples=400, n_features=2, centers=[[1, 0]], cluster_std=0.2, random_state=45)
    x1, y1 = make_blobs(n_samples=400, n_features=2, centers=[[2.25, 1.5]], cluster_std=0.2, random_state=45)
    x2, y2 = make_blobs(n_samples=400, n_features=2, centers=[[1.75, 2]], cluster_std=0.2, random_state=45)
    y1 += 1
    y2 += 2
    X = np.vstack((x0, x1, x2))
    y = np.hstack((y0, y1, y2))
    # # Visualize the test data
    # fig0, ax0 = plt.subplots()
    # for label in range(3):
    #     ax0.plot(X[y == label][:, 0], X[y == label][:, 1], '.',
    #              color=colors[label])
    # ax0.set_xlim(0.1, 3)
    # ax0.set_ylim(-0.75, 2.75)
    # ax0.set_title('Test data: 200 points x3 clusters.')
    return X, y


if __name__ == '__main__':
    X, y = _generateFig6()
    marker_size = 4
    dpi = 90
    fig_size = (8, 6)
    # plot ori data and save
    fig1 = plt.figure(figsize=fig_size, dpi=dpi, num=1)
    ax_fig1 = fig1.gca()
    ax_fig1.grid(True)
    for label in range(3):
        ax_fig1.plot(X[y == label][:, 0], X[y == label][:, 1], '.',
                     color=colors[label], markersize=marker_size, label="Cluster %d" % (label + 1))
    ax_fig1.set_xlim(0.1, 3)
    ax_fig1.set_ylim(-0.75, 2.75)
    lg = ax_fig1.legend(loc='upper left', fancybox=True, framealpha=0.5, prop={'size': 8})
    ax_fig1.set_title("Original Dataset")
    # create directory to save
    path=r".\video"
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    plt.savefig(r".\video\fig6_ori.png", dpi=dpi, bbox_inches='tight')
    # plot animation and save
    fig2 = plt.figure(figsize=fig_size, dpi=dpi, num=2)
    ax = fig2.gca()
    ax.grid(True)
    n_cluster, sigma_v, alpha_cut = 3, 3, 0.9
    n_cluster, sigma_v, alpha_cut = 3, 3, 0.5
    n_cluster, sigma_v, alpha_cut = 10, 3, 0.9
    n_cluster, sigma_v, alpha_cut = 10, 4, 0.65
    ini_save_name = r".\video\fig6_ini_%d.png" % n_cluster
    last_frame_name = r'.\video\fig6_n_%d_sigmav_%.1f_alpha_%.1f_last_frame.png' % (n_cluster, sigma_v, alpha_cut)
    tmp_video_name = r'.\video\fig6_n_%d_sigmav_%.1f_alpha_%.1f_tmp.mp4' % (n_cluster, sigma_v, alpha_cut)
    video_save_newFps_name = r'.\video\fig6_n_%d_sigmav_%.1f_alpha_%.1f.mp4' % (n_cluster, sigma_v, alpha_cut)
    clf = npcm(X, n_cluster, sigma_v, ax=ax, x_lim=(0.1, 3), y_lim=(-0.75, 2.75), alpha_cut=alpha_cut,
                  ini_save_name=ini_save_name, last_frame_name=last_frame_name)
    # we should set "blit=False,repeat=False" or the program would fail. "init_func=clf.init_animation" plot the
    # background of each frame There is not much point to use blit=True, if most parts of your plot should be
    # refreshed. see http://stackoverflow.com/questions/14844223/python-matplotlib-blit-to-axes-or-sides-of-the
    # -figure
    # To begin with, if you're chaining the ticks, etc, there isn't much point in using blitting. Blitting is
    #  just a way to avoid re-drawing everything if only some things are changing. If everything is changing,
    # there's no point in using blitting. Just re-draw the plot.
    anim = animation.FuncAnimation(fig2, clf, frames=clf.fit,
                                   init_func=clf.init_animation, interval=2000, blit=True, repeat=False)
    # anim.save(tmp_video_name, fps=1, extra_args=['-vcodec', 'libx264'], dpi='figure')
    # new_fps = 24
    # play_slow_rate = 1.5  # controls how many times a frame repeats.
    # movie_reader = FFMPEG_VideoReader(tmp_video_name)
    # movie_writer = FFMPEG_VideoWriter(video_save_newFps_name, movie_reader.size, new_fps)
    # print "n_frames:", movie_reader.nframes
    # # the 1st frame of the saved video can't be directly read by movie_reader.read_frame(), I don't know why
    # # maybe it's a bug of anim.save, actually, if we look at the movie we get from anim.save
    # # we can easilly see that the 1st frame just close very soon.
    # # so I manually get it at time 0, this is just a trick, I think.
    # tmp_frame = movie_reader.get_frame(0)
    # [movie_writer.write_frame(tmp_frame) for _ in range(int(new_fps * play_slow_rate))]
    # # for the above reason, we should read (movie_reader.nframes-1) frames so that the last frame is not
    # # read twice (not that get_frame(0) alread read once)
    # # However, I soon figure out that it should be (movie_reader.nframes-2). The details: we have actually
    # # 6 frames, but (print movie_reader.nframes) is 7. I read the first frame through movie_reader.get_frame(0)
    # # then are are 5 left. So I should use movie_reader.nframes - 2. Note that in fig1_pcm_fs2.py
    # # in the case of: original fps=1
    # # new_fps = 24, play_slow_rate = 1.5 the result is: 1st frame last 1.8s, others 1.5s, i.e., the 1st frame
    # # has more duration. This is messy.
    # for i in range(movie_reader.nframes - 2):
    #     tmp_frame = movie_reader.read_frame()
    #     [movie_writer.write_frame(tmp_frame) for _ in range(int(new_fps * play_slow_rate))]
    #     pass
    # movie_reader.close()
    # movie_writer.close()
    # os.remove(tmp_video_name)
    plt.show()
    pass
