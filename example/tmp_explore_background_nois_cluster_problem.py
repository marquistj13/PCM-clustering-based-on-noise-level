# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from algorithms import npcm_without_handle_backnoise
from sklearn.datasets import make_blobs

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
plt.style.use('classic')

from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import os

x_lim = (-0.5, 20.5)
y_lim = (-0.5, 10.5)


def _generateFig():
    """
    Two close clusters, one big and the other small,
    :return:
    """

    x0, y0 = make_blobs(n_samples=500, n_features=2, centers=[[15, 2.5]], cluster_std=1, random_state=45)
    x1, y1 = make_blobs(n_samples=500, n_features=2, centers=[[15, 7.5]], cluster_std=1, random_state=45)
    noise_x = np.random.uniform(0, 20, size=400)
    noise_y = np.random.uniform(0, 10, size=400)
    noise = np.r_['1,2,0', noise_x, noise_y]
    print noise.shape
    y1 += 1
    X = np.vstack((x0, x1))
    y = np.hstack((y0, y1))
    noise_label = np.zeros((len(noise))) + 2
    # # Visualize the test data
    # fig0, ax = plt.subplots()
    # for label in range(3):
    #     ax.plot(X[y == label][:, 0], X[y == label][:, 1], '.',
    #             color=colors[label])
    # ax.plot(noise_x, noise_y, '.', color=colors[3])
    # ax.set_xlim(x_lim)
    # ax.set_ylim(y_lim)
    # # ax0.set_title('Test data: 200 points x3 clusters.')
    X = np.vstack((X, noise))
    y = np.hstack((y, noise_label))
    return X, y


if __name__ == '__main__':
    # X, y = _generateFig()
    # np.savez(r".\video\explore_background_noise_cluster_data", X=X, y=y)
    tmp_file = np.load(r".\video\noise_cluster_problem_data.npz")
    X, y = tmp_file['X'], tmp_file['y']
    marker_size = 4
    dpi = 90
    fig_size = (7, 3.5)
    # plot ori data and save
    fig1 = plt.figure(figsize=fig_size, dpi=dpi, num=1)
    ax_fig1 = fig1.gca()
    ax_fig1.grid(True)
    for label in range(3):
        ax_fig1.plot(X[y == label][:, 0], X[y == label][:, 1], '.',
                     color=colors[label], markersize=marker_size, label="Cluster %d" % (label + 1))
    ax_fig1.set_xlim(x_lim)
    ax_fig1.set_ylim(y_lim)
    lg = ax_fig1.legend(loc='upper left', fancybox=True, framealpha=0.5, prop={'size': 8})
    ax_fig1.set_title("Original Dataset")
    plt.savefig(r".\video\explore_background_noise_cluster_ori.png", dpi=dpi, bbox_inches='tight')
    # plot animation and save
    fig2 = plt.figure(figsize=fig_size, dpi=dpi, num=2)
    ax = fig2.gca()
    ax.grid(True)
    n_cluster, sigma_v, alpha_cut = 10, 1, 0.1
    n_cluster, sigma_v, alpha_cut = 10, 0.5, 0.1
    n_cluster, sigma_v, alpha_cut = 10, 1, 0.5
    n_cluster, sigma_v, alpha_cut = 10, 1, 0.5
    ini_save_name = r".\video\explore_background_noise_cluster_ini.png"
    last_frame_name = r'.\video\explore_background_noise_cluster_last_frame.png'
    tmp_video_name = r'.\video\explore_background_noise_cluster_n_%d_sigmav_%.1f_alpha_%.1f_tmp.mp4' % (
        n_cluster, sigma_v, alpha_cut)
    video_save_newFps_name = r'.\video\explore_background_noise_cluster_n_%d_sigmav_%.1f_alpha_%.1f.mp4' % (
        n_cluster, sigma_v, alpha_cut)
    clf = npcm_without_handle_backnoise(X, n_cluster, ax=ax, x_lim=(x_lim), y_lim=(y_lim), alpha_cut=alpha_cut,
               ini_save_name=ini_save_name, last_frame_name=last_frame_name)

    anim = animation.FuncAnimation(fig2, clf, frames=clf.fit,
                                   init_func=clf.init_animation, interval=15, blit=True, repeat=False)
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
