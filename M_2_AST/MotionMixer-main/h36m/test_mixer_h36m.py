import torch
import os
from datasets.dataset_h36m import H36M_Dataset
from datasets.dataset_h36m_ang import H36M_Dataset_Angle
from utils.data_utils import define_actions
from torch.utils.data import DataLoader
from model import MlpMixer
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import argparse
from utils.utils_mixer import delta_2_gt, mpjpe_error, euler_error
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel


def test_pretrained(model, args):
    N = 1
    eval_frame = [1, 3, 7, 9, 13, 17, 21, 24]

    t_3d = np.zeros(len(eval_frame))

    t_3d_all = []
    t_3d_all_1 = []
    t_3d_all_2 = []
    t_3d_all_3 = []
    t_3d_all_4 = []
    t_3d_all_5 = []
    t_3d_all_6 = []
    t_3d_all_7 = []
    t_3d_all_8 = []
    device = args.dev

    accum_loss = [0, 0, 0, 0, 0, 0, 0, 0]
    n_batches = 0  # number of batches for all the sequences
    i = 0
    n = 0

    loss = [0, 0, 0, 0, 0, 0, 0, 0]
    T = [2, 4, 8, 10, 14, 18, 22, 25]
    accum_loss_ = 0

    model.eval()
    actions = define_actions(args.actions_to_consider)
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    idx_eval = 5

    m_p3d_h36_ = np.zeros([args.output_n])

    t_3d_all_a_1 = 0
    t_3d_all_a_2 = 0
    t_3d_all_a_3 = 0
    t_3d_all_a_4 = 0
    t_3d_all_a_5 = 0
    t_3d_all_a_6 = 0
    t_3d_all_a_7 = 0
    t_3d_all_a_8 = 0

    for action in actions:
        running_loss_ = 0
        running_loss = [0, 0, 0, 0, 0, 0, 0, 0]
        n = 0
        dataset_test = H36M_Dataset(args.data_dir, args.input_n, args.output_n, args.skip_rate, split=2,
                                    actions=[action])

        t_3_daction = np.zeros(len(eval_frame))
        t_3d_all_action = []
        t_3d_all_1_action = []
        t_3d_all_2_action = []
        t_3d_all_3_action = []
        t_3d_all_4_action = []
        t_3d_all_5_action = []
        t_3d_all_6_action = []
        t_3d_all_7_action = []
        t_3d_all_8_action = []




        test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0,
                                 pin_memory=True)
        for cnt, batch in enumerate(test_loader):
            with torch.no_grad():

                batch = batch.to(args.device)
                batch_dim = batch.shape[0]
                n += batch_dim

                all_joints_seq = batch.clone()[:, args.input_n:args.input_n + args.output_n, :]
                all_joints_seq_gt = batch.clone()[:, args.input_n:args.input_n + args.output_n, :]

                sequences_train = batch[:, 0:args.input_n, dim_used].view(-1, args.input_n, len(dim_used))

                sequences_gt = batch[:, args.input_n:args.input_n + args.output_n, dim_used].view(-1, args.output_n,
                                                                                                  args.pose_dim)

                if args.delta_x:
                    sequences_all = torch.cat((sequences_train, sequences_gt), 1)
                    sequences_all_delta = [sequences_all[:, 1, :] - sequences_all[:, 0, :]]
                    for i in range(args.input_n + args.output_n - 1):
                        sequences_all_delta.append(sequences_all[:, i + 1, :] - sequences_all[:, i, :])

                    sequences_all_delta = torch.stack((sequences_all_delta)).permute(1, 0, 2)
                    sequences_train_delta = sequences_all_delta[:, 0:args.input_n, :]
                    sequences_predict = model(sequences_train_delta)
                    sequences_predict = delta_2_gt(sequences_predict, sequences_train[:, -1, :])

                    sequences_gt_3d = sequences_gt.reshape(sequences_gt.shape[0], sequences_gt.shape[1], -1, 3)
                    sequences_predict_3d = sequences_predict.reshape(sequences_predict.shape[0],
                                                                     sequences_predict.shape[1], -1, 3)
                    # print(sequences_predict_3d.shape,'sequences_predict_3d', sequences_predict_3d.shape,'sequences_predict_3d')
                    # torch.Size([256, 25, 22, 3]) sequences_predict_3d torch.Size([256, 25, 22, 3]) sequences_predict_3d

                    # for k in np.arange(0, len(eval_frame)):
                    #       j = eval_frame[k]
                    #       t_3d[k] += torch.mean(torch.norm(sequences_gt_3d[:, j, :, :].contiguous().view(-1, 3) - sequences_predict_3d[:, j, :, :].contiguous().view(-1, 3), 2, 1)).item() * n

                    N += n

                else:
                    sequences_predict = model(sequences_train)
                    loss = mpjpe_error(sequences_predict, sequences_gt)

                all_joints_seq[:, :, dim_used] = sequences_predict
                all_joints_seq[:, :, index_to_ignore] = all_joints_seq[:, :, index_to_equal]

                all_joints_seq_gt[:, :, dim_used] = sequences_gt
                all_joints_seq_gt[:, :, index_to_ignore] = all_joints_seq_gt[:, :, index_to_equal]

                # all_joints_seq_gt.shape ([256, 25, 96])

                # 包涵32 全骨骼 不忽略的
                all_joints_seq_3d = all_joints_seq.view(sequences_predict.shape[0], sequences_predict.shape[1], -1, 3)
                all_joints_seq_gt_3d = all_joints_seq_gt.view(sequences_gt.shape[0], sequences_gt.shape[1], -1, 3)

                for k in np.arange(0, len(eval_frame)):
                    j = eval_frame[k]
                    # t_3d[k] += torch.mean(torch.norm(all_joints_seq_gt_3d[:, j, :, :].contiguous().view(-1, 3) - all_joints_seq_3d[:, j, :, :].contiguous().view(-1, 3), 2, 1)).item() * n
                    t_3d[k] += torch.sum(torch.mean(torch.norm(
                        all_joints_seq_gt_3d[:, j, :, :].contiguous() - all_joints_seq_3d[:, j, :, :].contiguous(),
                        dim=2), dim=1), dim=0).item()
                    t_3_daction[k] += torch.sum(torch.mean(torch.norm(
                        all_joints_seq_gt_3d[:, j, :, :].contiguous() - all_joints_seq_3d[:, j, :, :].contiguous(),
                        dim=2), dim=1), dim=0).item()

                t_3d_all_1_action.append(t_3_daction[0] / n)
                t_3d_all_2_action.append(t_3_daction[1] / n)
                t_3d_all_3_action.append(t_3_daction[2] / n)
                t_3d_all_4_action.append(t_3_daction[3] / n)
                t_3d_all_5_action.append(t_3_daction[4] / n)
                t_3d_all_6_action.append(t_3_daction[5] / n)
                t_3d_all_7_action.append(t_3_daction[6] / n)
                t_3d_all_8_action.append(t_3_daction[7] / n)

                # 数据pjpe 代码计算 */******************************/*

                mpjpe_h36_ = torch.sum(torch.mean(torch.norm(
                    all_joints_seq.view(-1, args.output_n, 32, 3) - all_joints_seq_gt.view(-1, args.output_n, 32, 3),
                    dim=3), dim=2), dim=0)
                m_p3d_h36_ += mpjpe_h36_.cpu().numpy()

                # a = all_joints_seq.view(-1,args.output_n,32,3)
                # b = all_joints_seq_gt.view(-1,args.output_n,32,3)
                # a = a[:, :25, :, :]
                # b = b[:, :25, :, :]
                # loss=mpjpe_error(a,b)
                # # loss=mpjpe_error(all_joints_seq.view(-1,args.output_n,32,3),all_joints_seq_gt.view(-1,args.output_n,32,3))

                a = all_joints_seq.view(-1, args.output_n, 32, 3)
                b = all_joints_seq_gt.view(-1, args.output_n, 32, 3)

                for j in range(8):
                    # T = [2, 4, 8, 10, 14, 18, 22, 25]
                    i = T[j]
                    a_ = a[:, :i, :, :]
                    b_ = b[:, :i, :, :]
                    t = mpjpe_error(a_, b_)

                    t_list = t.cpu().numpy().tolist()

                    loss[j] = t_list

                    # loss=mpjpe_error(all_joints_seq,sequences_predict_gt) # loss in milimeters
                    accum_loss[j] += loss[j] * batch_dim
                    running_loss[j] += loss[j] * batch_dim

                t = mpjpe_error(a, b)

                running_loss_ += t * batch_dim
                accum_loss_ += t * batch_dim

                t_1 = a.cpu().numpy()
                t_2 = b.cpu().numpy()
                N_1 = t_1[2, :, :]
                N_2 = t_2[2, :, :]
                actions_name = 't_1' + str(action)
                actions_name_ = 't_2' + str(action)

        print(' ******************************* ' + str(action) + ' ******************************* ')
        print(str(action) + ' 80mm t_3d loss in mm is: ', np.mean(t_3d_all_1_action))
        print(str(action) + ' 160mm t_3d loss in mm is: ', np.mean(t_3d_all_2_action))
        print(str(action) + ' 320mm t_3d loss in mm is: ', np.mean(t_3d_all_3_action))
        print(str(action) + ' 400mm t_3d loss in mm is: ', np.mean(t_3d_all_4_action))
        print(str(action) + ' 560mm t_3d loss in mm is: ', np.mean(t_3d_all_5_action))
        print(str(action) + ' 720mm t_3d loss in mm is: ', np.mean(t_3d_all_6_action))
        print(str(action) + ' 880mm t_3d loss in mm is: ', np.mean(t_3d_all_7_action))
        print(str(action) + ' 1000mm t_3d loss in mm is: ', np.mean(t_3d_all_8_action))

        t_3d_all_a_1 += np.mean(t_3d_all_1_action)
        t_3d_all_a_2 += np.mean(t_3d_all_2_action)
        t_3d_all_a_3 += np.mean(t_3d_all_3_action)
        t_3d_all_a_4 += np.mean(t_3d_all_4_action)
        t_3d_all_a_5 += np.mean(t_3d_all_5_action)
        t_3d_all_a_6 += np.mean(t_3d_all_6_action)
        t_3d_all_a_7 += np.mean(t_3d_all_7_action)
        t_3d_all_a_8 += np.mean(t_3d_all_8_action)

        print('loss at test subject for action : ' + str(action) + ' is: ' + str(running_loss_ / n))

        for j in range(8):
            item = running_loss[j]
            print(str(action) + ' overall loss in mm is: ' + str(item / n))

        print('\n')
        n_batches += n

        # print('xx: ',np.mean(t_3d[idx_eval]/N))
        t_3d_all_1.append(t_3d[0] / N)
        t_3d_all_2.append(t_3d[1] / N)
        t_3d_all_3.append(t_3d[2] / N)
        t_3d_all_4.append(t_3d[3] / N)
        t_3d_all_5.append(t_3d[4] / N)
        t_3d_all_6.append(t_3d[5] / N)
        t_3d_all_7.append(t_3d[6] / N)
        t_3d_all_8.append(t_3d[7] / N)

    m_p3d_h36_ = m_p3d_h36_ / n_batches
    print(m_p3d_h36_.shape)
    print(' ******************************* ' + 'Average' + ' ******************************* ')

    print('overall average loss in mm is: ' + str(accum_loss_ / n_batches))

    for j in range(8):
        item = accum_loss[j]
        print('overall loss in mm is: ' + str(item / n_batches))

    titles = np.array(range(args.output_n)) + 1
    ret = {}
    results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']
    for j in range(args.output_n):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36_[j], m_p3d_h36_[j]]
    print([round(ret[key][0], 1) for key in results_keys])  # 奔驰

    print('overall 80mm loss in mm is: ', np.mean(t_3d_all_1))
    print('overall 160mm loss in mm is: ', np.mean(t_3d_all_2))
    print('overall 320mm loss in mm is: ', np.mean(t_3d_all_3))
    print('overall 400mm loss in mm is: ', np.mean(t_3d_all_4))
    print('overall 560mm loss in mm is: ', np.mean(t_3d_all_5))
    print('overall 720mm loss in mm is: ', np.mean(t_3d_all_6))
    print('overall 880mm loss in mm is: ', np.mean(t_3d_all_7))
    print('overall 1000mm loss in mm is: ', np.mean(t_3d_all_8))

    t_3d_all_a_1 = t_3d_all_a_1/15
    t_3d_all_a_2 = t_3d_all_a_2/15
    t_3d_all_a_3 = t_3d_all_a_3/15
    t_3d_all_a_4 = t_3d_all_a_4/15
    t_3d_all_a_5 = t_3d_all_a_5/15
    t_3d_all_a_6 = t_3d_all_a_6/15
    t_3d_all_a_7 = t_3d_all_a_7/15
    t_3d_all_a_8 = t_3d_all_a_8/15

    print('mean 80mm loss in mm is: ', t_3d_all_a_1)
    print('mean 160mm loss in mm is: ', t_3d_all_a_2)
    print('mean 320mm loss in mm is: ', t_3d_all_a_3)
    print('mean 400mm loss in mm is: ', t_3d_all_a_4)
    print('mean 560mm loss in mm is: ', t_3d_all_a_5)
    print('mean 720mm loss in mm is: ', t_3d_all_a_6)
    print('mean 880mm loss in mm is: ', t_3d_all_a_7)
    print('mean 1000mm loss in mm is: ',t_3d_all_a_8)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False) # Parameters for mpjpe
    parser.add_argument('--data_dir', type=str, default=r'/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/h36m/data_h36m', help='path to the unziped dataset directories(H36m/AMASS/3DPW)')  
    parser.add_argument('--input_n', type=int, default=10, help="number of model's input frames")
    parser.add_argument('--output_n', type=int, default=25, help="number of model's output frames")
    parser.add_argument('--skip_rate', type=int, default=1, choices=[1, 5], help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers in the dataloader')
    parser.add_argument('--root', default=r'/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/h36m/test_run', type=str, help='root path for the logging') #'./runs'

    parser.add_argument('--activation', default='mish', type=str, required=False)  # 'mish', 'gelu'
    parser.add_argument('--r_se', default=8, type=int, required=False)

    parser.add_argument('--n_epochs', default=25, type=int, required=False)
    parser.add_argument('--batch_size', default=50, type=int, required=False)  # 100  50  in all original 50
    parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
    parser.add_argument('--pin_memory', default=False, type=bool, required=False)
    parser.add_argument('--loader_workers', default=4, type=int, required=False)
    parser.add_argument('--load_checkpoint', default=False, type=bool, required=False)
    parser.add_argument('--dev', default='cuda:0', type=str, required=False)
    parser.add_argument('--initialization', type=str, default='none', help='none, glorot_normal, glorot_uniform, hee_normal, hee_uniform')
    parser.add_argument('--use_scheduler', default=True, type=bool, required=False)
    parser.add_argument('--milestones', type=list, default=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90], help='the epochs after which the learning rate is adjusted by gamma')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma correction to the learning rate, after reaching the milestone epochs')
    parser.add_argument('--clip_grad', type=float, default=None, help='select max norm to clip gradients')
    parser.add_argument('--model_path', type=str, default=r'/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/checkpoints/hm36/hm36_3d_25frames_ckpt_epoch_9', help='directory with the models checkpoints ')
    parser.add_argument('--actions_to_consider', default='all', help='Actions to visualize.Choose either all or a list of actions')
    parser.add_argument('--batch_size_test', type=int, default=256, help='batch size for the test set')
    parser.add_argument('--visualize_from', type=str, default='test', choices=['train', 'val', 'test'], help='choose data split to visualize from(train-val-test)')
    parser.add_argument('--loss_type', type=str, default='mpjpe', choices=['mpjpe', 'angle'])
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'])
         

    
    

    args = parser.parse_args()

    if args.loss_type == 'mpjpe':
        parser_mpjpe = argparse.ArgumentParser(parents=[parser]) # Parameters for mpjpe
        parser_mpjpe.add_argument('--hidden_dim', default=66, type=int, required=False)  
        parser_mpjpe.add_argument('--num_blocks', default=4, type=int, required=False)  
        parser_mpjpe.add_argument('--tokens_mlp_dim', default=20, type=int, required=False)
        parser_mpjpe.add_argument('--channels_mlp_dim', default=66, type=int, required=False)  
        parser_mpjpe.add_argument('--regularization', default=0.1, type=float, required=False)  
        parser_mpjpe.add_argument('--pose_dim', default=66, type=int, required=False)
        parser_mpjpe.add_argument('--delta_x', type=bool, default=True, help='predicting the difference between 2 frames')
        parser_mpjpe.add_argument('--lr', default=0.001, type=float, required=False)  
        args = parser_mpjpe.parse_args()
    
    elif args.loss_type == 'angle':
        parser_angle = argparse.ArgumentParser(parents=[parser]) # Parameters for angle
        parser_angle.add_argument('--hidden_dim', default=60, type=int, required=False) 
        parser_angle.add_argument('--num_blocks', default=3, type=int, required=False) 
        parser_angle.add_argument('--tokens_mlp_dim', default=40, type=int, required=False)
        parser_angle.add_argument('--channels_mlp_dim', default=60, type=int, required=False) 
        parser_angle.add_argument('--regularization', default=0.0, type=float, required=False)
        parser_angle.add_argument('--pose_dim', default=48, type=int, required=False)
        parser_angle.add_argument('--lr', default=1e-02, type=float, required=False) 
        args = parser_angle.parse_args()
    
    if args.loss_type == 'angle' and args.delta_x:
        raise ValueError('Delta_x and loss type angle cant be used together.')

    print(args)

    model = MlpMixer(num_classes=args.pose_dim, num_blocks=args.num_blocks,
                     hidden_dim=args.hidden_dim, tokens_mlp_dim=args.tokens_mlp_dim,
                     channels_mlp_dim=args.channels_mlp_dim, seq_len=args.input_n,
                     pred_len=args.output_n, activation=args.activation,
                     mlp_block_type='normal', regularization=args.regularization,
                     input_size=args.pose_dim, initialization='none', r_se=args.r_se,
                     use_max_pooling=False, use_se=True)

    model = model.to(args.dev)
    # model = DataParallel(model)
    checkpoint = torch.load(args.model_path)  # 加载断点

    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    # model.load_state_dict(torch.load(args.model_path))
    
    print('total number of parameters of the network is: ' +
          str(sum(p.numel() for p in model.parameters() if p.requires_grad)))



    
    test_pretrained(model, args)



