import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.autograd
import matplotlib.pyplot as plt
from utils.ang2joint import *
from dataloader_3dpw_eval import *
# from dataloader_amass import *
import numpy as np
import argparse
import os
from mlp_mixer import MlpMixer


def test_mixer(model, args):

    device = args.dev
    model.eval()
    accum_loss = [0,0,0,0,0,0,0,0]
    n_batches = 0  # number of batches for all the sequences
    i = 0
    n = 0
    loss = [0,0,0,0,0,0,0,0]
    T = [2, 4, 8, 10, 14, 18, 22, 25]
    accum_loss_ = 0
    
    Dataset = Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate,split=0)
    loader_test = DataLoader( Dataset,batch_size=args.batch_size,
                              shuffle =False,num_workers=0)
        
                      
    joint_used=np.arange(4,22)  #[4,5.....21] 18 个数
    full_joint_used=np.arange(0,22) # needed for visualization
    with torch.no_grad():
        for cnt,batch in enumerate(loader_test):

            batch = batch.float().to(device)
            batch_dim=batch.shape[0]
            n+=batch_dim
            
            sequences_train=batch[:,0:args.input_n,joint_used,:].view(-1,args.input_n,args.pose_dim)
    
            sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,full_joint_used,:]#.view(-1,args.output_n,args.pose_dim)
            
            sequences_predict=model(sequences_train).view(-1,args.output_n,18,3)#.permute(0,1,3,2)
            
            
            all_joints_seq=sequences_predict_gt.clone()
    
            all_joints_seq[:,:,joint_used,:]=sequences_predict
            a = all_joints_seq
            b = sequences_predict_gt
           
            for j in range(8):
            # T = [2, 4, 8, 10, 14, 18, 22, 25]
              i = T[j]
              a_ = a[:, :i, :, :]
              b_ = b[:, :i, :, :]
              t = mpjpe_error(a_,b_)*1000
              t_list = t.cpu().numpy().tolist()
             
              loss[j]=t_list

              # loss=mpjpe_error(all_joints_seq,sequences_predict_gt)*1000 # loss in milimeters
              accum_loss[j]+=loss[j]*batch_dim
              accum_loss_+=t * batch_dim
    for j in range(8):       
      item = accum_loss[j] 
      print('overall loss in mm is: '+str(item/n))



    return accum_loss_/n_batches



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
    parser = argparse.ArgumentParser(add_help=False) # Parameters for mpjpe
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/amass/AMASS_dataset', help='path to the unziped dataset directories(H36m/AMASS/3DPW)')
    parser.add_argument('--input_n', type=int, default=10, help="number of model's input frames")
    parser.add_argument('--output_n', type=int, default=25, help="number of model's output frames")
    parser.add_argument('--skip_rate', type=int, default=5, choices=[1, 5], help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers in the dataloader')
    parser.add_argument('--root', default='/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/amass/run_for_longging', type=str, help='root path for the logging')

    parser.add_argument('--activation', default='gelu', type=str, required=False) 
    parser.add_argument('--r_se', default=8, type=int, required=False)

    parser.add_argument('--n_epochs', default=50, type=int, required=False)
    parser.add_argument('--batch_size', default=50, type=int, required=False) 
    parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
    parser.add_argument('--pin_memory', default=False, type=bool, required=False)
    parser.add_argument('--loader_workers', default=4, type=int, required=False)
    parser.add_argument('--load_checkpoint', default=False, type=bool, required=False)
    parser.add_argument('--dev', default='cuda:0', type=str, required=False)
    parser.add_argument('--initialization', type=str, default='none', help='none, glorot_normal, glorot_uniform, hee_normal, hee_uniform')
    parser.add_argument('--use_scheduler', default=True, type=bool, required=False)
    parser.add_argument('--milestones', type=list, default=[15, 25, 35, 40], help='the epochs after which the learning rate is adjusted by gamma')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma correction to the learning rate, after reaching the milestone epochs')
    parser.add_argument('--clip_grad', type=float, default=None, help='select max norm to clip gradients')
    parser.add_argument('--model_path', type=str, default='/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/checkpoints/amass/amass_3d_25frames_ckpt_epoch_26', help='directory with the models checkpoints ')
    parser.add_argument('--batch_size_test', type=int, default=256, help='batch size for the test set')
    parser.add_argument('--visualize_from', type=str, default='test', choices=['train', 'val', 'test'], help='choose data split to visualize from(train-val-test)')

    args = parser.parse_args()

    parser_mpjpe = argparse.ArgumentParser(parents=[parser]) # Parameters for mpjpe
    parser_mpjpe.add_argument('--hidden_dim', default=128, type=int, required=False)  
    parser_mpjpe.add_argument('--num_blocks', default=4, type=int, required=False)
    parser_mpjpe.add_argument('--tokens_mlp_dim', default=20, type=int, required=False)
    parser_mpjpe.add_argument('--channels_mlp_dim', default=128, type=int, required=False)  
    parser_mpjpe.add_argument('--regularization', default=0.1, type=float, required=False) 
    parser_mpjpe.add_argument('--pose_dim', default=54, type=int, required=False)
    parser_mpjpe.add_argument('--delta_x', type=bool, default=True, help='predicting the difference between 2 frames')
    parser_mpjpe.add_argument('--lr', default=0.001, type=float, required=False)  
    args = parser_mpjpe.parse_args()
    


    # print(args)

    model = MlpMixer(num_classes=args.pose_dim, num_blocks=args.num_blocks,
                     hidden_dim=args.hidden_dim, tokens_mlp_dim=args.tokens_mlp_dim,
                     channels_mlp_dim=args.channels_mlp_dim, seq_len=args.input_n,
                     pred_len=args.output_n, activation=args.activation,
                     mlp_block_type='normal', regularization=args.regularization,
                     input_size=args.pose_dim, initialization='none', r_se=args.r_se,
                     use_max_pooling=False, use_se=True)

    model = model.to(args.dev)

   
    checkpoint = torch.load(args.model_path)  # 加载断点

    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    # model.load_state_dict(torch.load(args.model_path))
    
    
    model.eval ()
    
   
    test_mixer(model, args)
   
  