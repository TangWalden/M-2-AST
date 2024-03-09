import time
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

class plot_h36m(object):

    def __init__(self, prediction_data, GT_data):
        print ('prediction_data',type(prediction_data))
        print ('GT_data',type(GT_data))
        self.joint_xyz = GT_data
        self.nframes = 10

        self.joint_xyz_f = prediction_data

        # set up the axes
        xmin = -800
        xmax = 800
        ymin = -800
        ymax = 800
        zmin = -800
        zmax = 800

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        self.ax.view_init(elev=30, azim=-80)
        # plt.axis('off')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        self.chain = [
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 5, 6, 7, 8]),
            np.array([0, 10, 11, 12, 13]),
            np.array([11, 14, 15, 16, 17]),
            np.array([11, 18, 19, 20, 21]),

                      # np.array([0, 1, 2, 3,]),17
                      # np.array([0, 4, 5, 6])
                      # ,
                      # np.array([0, 7, 8, 9, 10]),
                      # np.array([8, 11, 12, 13]),
                      # np.array([8, 14, 15, 16])
                      ]
        print (type(self.chain))
        self.scats = []
        self.lns = []

    def update(self, frame):
        for scat in self.scats:
            scat.remove()
        for ln in self.lns:
            self.ax.lines.pop(0)  # 删除第一个元素
        self.scats = []
        self.lns = []

        xdata = np.squeeze(self.joint_xyz[frame, :, 0])
        ydata = np.squeeze(self.joint_xyz[frame, :, 1])
        zdata = np.squeeze(self.joint_xyz[frame, :, 2])
        xdata_f = np.squeeze(self.joint_xyz_f[frame, :, 0])
        ydata_f = np.squeeze(self.joint_xyz_f[frame, :, 1])
        zdata_f = np.squeeze(self.joint_xyz_f[frame, :, 2])

        """
        在机器学习和深度学习中，通常算法的结果是可以表示向量的数组（即包含两对或以上的方括号形式[[]]），
        如果直接利用这个数组进行画图可能显示界面为空（见后面的示例）。
        我们可以利用squeeze（）函数将表示向量的数组转换为秩为1的数组，这样利用matplotlib库函数画图时，就可以正常的显示结果了。
        """

        for i in range(len(self.chain)):
            # self.lns.append(self.ax.plot3D(xdata_f[self.chain[i][:],], ydata_f[self.chain[i][:],], zdata_f[self.chain[i][:],], linewidth=2.0, color='#FFA500')) # orange: prediction
            self.lns.append(self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][:],], zdata[self.chain[i][:],], linewidth=2.0, color='k')) # black: ground truth
    def plot(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40, repeat=True)
        # ani.save(f'{base_path}/test_1.gif', writer='pillow')
        plt.show()





































# if __name__ == '__main__':
#     # config = yaml.load(open('config.yml'),Loader=yaml.FullLoader)
#     # use_node = np.array([0,1,2,3,6,7,8,11,12,13,14,15,16,17,20,21,21])
#     use_node = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
#     print(len(use_node))
#     #load GT_data
#     # base_path = config['base_dir_human36']
#     base_path = r"E:\数据处理\siMLPe\exps\baseline_h36m\pre data & data"
#
#     for actions in ['walktogether']:
#         # test_save_path = os.path.join(f'{base_path}{actions}', 'test.npy')
#         test_save_path = r"E:\数据处理\siMLPe\exps\baseline_h36m\pre data & data\test_data\data1.npy"
#         GT_data = np.load(test_save_path)
#
#         # load prediction_data
#         # prediction_data_path = os.path.join(f'{base_path}{actions}', 'vis.npy')
#         prediction_data_path = r"E:\数据处理\siMLPe\exps\baseline_h36m\pre data & data\test_data\data1.npy"
#         prediction_data = np.load(prediction_data_path)
#
#
#         print('prediction_data:\n',prediction_data.shape)
#         print('GT_data:\n',GT_data.shape)
#
#         prediction_data = prediction_data[:,use_node,:]
#         GT_data = GT_data[:,use_node,:]
#
#
#         predict_plot = plot_h36m(prediction_data, GT_data)
#         predict_plot.plot()




if __name__ == '__main__':
    # config = yaml.load(open('config.yml'),Loader=yaml.FullLoader)
    # use_node = np.array([0,1,2,3,6,7,8,11,12,13,14,15,16,17,20,21,21])
    use_node = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    print(len(use_node))
    #load GT_data
    # base_path = config['base_dir_human36']
    base_path = r"E:\数据处理\siMLPe\exps\baseline_h36m\pre data & data"

    for actions in ['walktogether']:
        # test_save_path = os.path.join(f'{base_path}{actions}', 'test.npy')
        test_save_path = r"E:\数据处理\siMLPe\exps\baseline_h36m\pre data & data\test_data\data1_prediction.npy"
        GT_data = np.load(test_save_path)

        # load prediction_data
        # prediction_data_path = os.path.join(f'{base_path}{actions}', 'vis.npy')
        prediction_data_path = r"E:\数据处理\siMLPe\exps\baseline_h36m\pre data & data\test_data\data1_prediction.npy"
        prediction_data = np.load(prediction_data_path)


        print('prediction_data:\n',prediction_data.shape)
        print('GT_data:\n',GT_data.shape)

        prediction_data = prediction_data[:,use_node,:]
        GT_data = GT_data[:,use_node,:]


        predict_plot = plot_h36m(prediction_data, GT_data)
        predict_plot.plot()



#
#
#
#
# #
# #
# #
# #
# #
# #
#
#
#
#
#
#
#
#
#
#
#
#


