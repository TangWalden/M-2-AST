from torch.utils.data import Dataset, DataLoader
import numpy as np
# from h5py import File
# import scipy.io as sio
from matplotlib import pyplot as plt
import torch
import os
from scipy.spatial.transform import Rotation as R
from utils.ang2joint import *
import networkx as nx
import pickle as pkl

'''
adapted from
https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/amass3d.py
'''


class Datasets(Dataset):

    def __init__(self, data_dir, input_n, output_n, skip_rate, actions=None, split=0):

        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = '/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/amass/AMASS_dataset/'
        # os.path.join(data_dir,'AMASS')         #  "D:\data\AMASS\\"
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        # self.sample_rate = opt.sample_rate
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)  # start from 4 for 17 joints, removing the non moving ones
        seq_len = self.in_n + self.out_n

        amass_splits = [
            ['3dpw_test'],
        ]

        # amass_splits = [
        #     ['ACCAD'],
        #     ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        #     ['BioMotionLab_NTroje'],
        # ]

        # amass_splits = [
        #     ['CMU'],
        #     ['SFU'],
        #     # ['BioMotionLab_NTroje'],
        #     ['3dpw_test'],
        # ]

        # amass_splits = [
        #     ['CMU'],
        #     ['CMU'],
        #     ['CMU']
        #     #['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        #     #['BioMotionLab_NTroje'],
        # ]

        # amass_splits = [['BioMotionLab_NTroje'], ['HumanEva'], ['SSM_synced']]
        # amass_splits = [['HumanEva'], ['HumanEva'], ['HumanEva']]
        # amass_splits[0] = list(
        #     set(amass_splits[0]).difference(set(amass_splits[1] + amass_splits[2])))

        # from human_body_prior.body_model.body_model import BodyModel
        # from smplx import lbs
        # root_path = os.path.dirname(__file__)
        # bm_path = root_path[:-6] + '/body_models/smplh/neutral/model.npz'
        # bm = BodyModel(bm_path=bm_path, num_betas=16, batch_size=1, model_type='smplh')
        # beta_mean = np.array([0.41771687, 0.25984767, 0.20500051, 0.13503872, 0.25965645, -2.10198147, -0.11915666,
        #                       -0.5498772, 0.30885323, 1.4813145, -0.60987528, 1.42565269, 2.45862726, 0.23001716,
        #                       -0.64180912, 0.30231911])
        # beta_mean = torch.from_numpy(beta_mean).unsqueeze(0).float()
        # # Add shape contribution
        # v_shaped = bm.v_template + lbs.blend_shapes(beta_mean, bm.shapedirs)
        # # Get the joints
        # # NxJx3 array
        # p3d0 = lbs.vertices2joints(bm.J_regressor, v_shaped)  # [1,52,3]
        # p3d0 = (p3d0 - p3d0[:, 0:1, :]).float().cuda().cpu().data.numpy()
        # parents = bm.kintree_table.data.numpy()[0, :]
        # np.savez_compressed('smpl_skeleton.npz', p3d0=p3d0, parents=parents)

        # load mean skeleton
        skel = np.load('/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/utils/body_models/smpl_skeleton.npz')
        self.p3d0 = torch.from_numpy(skel['p3d0']).float()[:, :22]  # torch.Size([1, 52, 3])
        parents = skel['parents']  # (52,)
        self.parent = {}
        for i in range(len(parents)):
            if i > 21:
                break
            self.parent[i] = parents[i]
        n = 0
        for ds in amass_splits[split]:
            print(self.path_to_data + ds)
            if not os.path.isdir(self.path_to_data + ds):
                print(ds)
                continue
            print('>>> loading {}'.format(ds))
            for sub in os.listdir(self.path_to_data + ds):
                print ("working in ",self.path_to_data + ds)

                if not os.path.isdir(self.path_to_data + ds + '/' + sub):
                    continue
                for act in os.listdir(self.path_to_data + ds + '/' + sub):
                    # print ("poses path",self.path_to_data + ds + '/' + sub + '/' + act)

                    # print (act)
                    # if not act.endswith('.npz'):
                    #     continue
                    # if not ('walk' in act or 'jog' in act or 'run' in act or 'treadmill' in act):
                    #     continue

                    pose_all = pkl.load(open(self.path_to_data + ds + '/' + sub + '/' + act, 'rb'), encoding='latin1')
                    # pose_all['trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses']


                    try:
                        poses_60Hz = pose_all['poses_60Hz']
                    except:
                        print('no poses_60Hz at {}_{}_{}'.format(ds, sub, act))
                        continue
                    frame_rate = 60

                    # gender = pose_all['gender']
                    # dmpls = pose_all['dmpls']
                    # betas = pose_all['betas']
                    # trans = pose_all['trans']
                    sample_rate = int(frame_rate // 25)
                    #  poses_60Hz (2, 1796, 72)

                    for i in range(len(poses_60Hz)):  # 2 2 2 ...1 2...

                        N = len(poses_60Hz[i])
                        # poses_60Hz = np.array(poses_60Hz)  # (2, 1796, 72)


                        fidxs = np.arange(0, N, sample_rate)  # 降采样
                        fn = len(fidxs)  # T
                        poses = poses_60Hz[i][fidxs]
                        # print(poses.shape,'0909')  # (898, 72)


                        # poses = torch.from_numpy(poses).float().cuda()

                        poses = poses.reshape(fn, -1, 3)  # (898, 24, 3)

                        poses = poses[:, :-2]

                        poses = R.from_rotvec(poses.reshape(-1, 3)).as_rotvec()

                        poses = poses.reshape(fn, 22, 3)
                        poses[:, 0] = 0

                        p3d0_tmp = self.p3d0.repeat([poses.shape[0], 1, 1])  # [1, 52, 3] -- [898, 52, 3]

                        p3d = ang2joint(p3d0_tmp, torch.tensor(poses).float(), self.parent)

                    # self.p3d[(ds, sub, act)] = p3d.cpu().data.numpy()
                        self.p3d.append(p3d.cpu().data.numpy())  # (1, 898, 22, 3)


                        valid_frames = np.arange(0, fn - seq_len + 1, skip_rate)

                        # tmp_data_idx_1 = [(ds, sub, act)] * len(valid_frames)
                        # print((ds, sub, act))  ('3dpw_test', 'test', 'office_phoneCall_00.pkl')
                        self.keys.append((ds, sub, act))
                        tmp_data_idx_1 = [n] * len(valid_frames)

                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))  # (173, 2) (346, 2)...(7834, 2)
                        # self.data_idx = np.array(self.data_idx)
                        # print(self.data_idx.shape)
                        # self.data_idx = self.data_idx.tolist()
                        # key, start_frame = self.data_idx[1]
                        # print(  self.data_idx)

                        n += 1  # n-0...36

    def __len__(self):
        return np.shape(self.data_idx)[0]  # 7834 长度为这个

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)

        return self.p3d[key][fs]  # , key


# In[12]:


def normalize_A(
        A):  # given an adj.matrix, normalize it by multiplying left and right with the degree matrix, in the -1/2 power

    A = A + np.eye(A.shape[0])

    D = np.sum(A, axis=0)

    D = np.diag(D.A1)

    D_inv = D ** -0.5
    D_inv[D_inv == np.infty] = 0

    return D_inv * A * D_inv


# In[ ]:


def spatio_temporal_graph(joints_to_consider, temporal_kernel_size,
                          spatial_adjacency_matrix):  # given a normalized spatial adj.matrix,creates a spatio-temporal adj.matrix

    number_of_joints = joints_to_consider

    spatio_temporal_adj = np.zeros((temporal_kernel_size, number_of_joints, number_of_joints))
    for t in range(temporal_kernel_size):
        for i in range(number_of_joints):
            spatio_temporal_adj[t, i, i] = 1  # create edge between same body joint,for t consecutive frames
            for j in range(number_of_joints):
                if spatial_adjacency_matrix[i, j] != 0:  # if the body joints are connected
                    spatio_temporal_adj[t, i, j] = spatial_adjacency_matrix[i, j]
    return spatio_temporal_adj




# In[23]:


def mpjpe_error(batch_pred, batch_gt):
    # assert batch_pred.requires_grad==True
    # assert batch_gt.requires_grad==False

    batch_pred = batch_pred.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)

    return torch.mean(torch.norm(batch_gt - batch_pred, 2, 1))


if __name__ == '__main__':
    data_dir = ''
    dataset = Datasets(data_dir, input_n=10,  # 取出数据集h36M
                       output_n=25, skip_rate=5, split=0)

    # print([0]*123)
    # pose_all = np.load(r'E:\MotionMixer-main-source\amass\AMASS_daraset\CMU\12\12_01_poses.npz')
    # print(pose_all.files)