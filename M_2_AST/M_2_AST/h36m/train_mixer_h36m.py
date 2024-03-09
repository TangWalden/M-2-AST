import torch
import os
from datasets.dataset_h36m import H36M_Dataset
from datasets.dataset_h36m_ang import H36M_Dataset_Angle
from utils.data_utils import define_actions
from torch.utils.data import DataLoader
from model import MlpMixer
import torch.optim as optim
import numpy as np
import argparse
from utils.utils_mixer import delta_2_gt, mpjpe_error, euler_error
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random

# seed
seed = 300
random.seed(seed)  # random
np.random.seed(seed)  # numpy
torch.manual_seed(seed)  # torch+CPU
torch.cuda.manual_seed(seed)  # torch+GPU

save_loss = 0
q = 0

def get_log_dir(out_dir):
    dirs = [x[0] for x in os.walk(out_dir)]
    if len(dirs ) < 2:
        log_dir = os.path.join(out_dir, 'exp0')
        os.mkdir(log_dir)
    else:
        log_dir = os.path.join(out_dir, 'exp%i'%(len(dirs)-1))
        os.mkdir(log_dir)

    return log_dir

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def train(model, model_name, args):

    log_dir = get_log_dir(args.root)
    tb_writer = SummaryWriter(log_dir=log_dir)
    print('Save data of the run in: %s'%log_dir)

    device = args.dev

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)

    if args.use_scheduler: # True
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma)

    train_loss, val_loss, test_loss = [], [], []

    if args.loss_type == 'mpjpe':  # loss 是mpjpe
        dataset = H36M_Dataset(args.data_dir, args.input_n,  # 取出数据集h36M
                        args.output_n, args.skip_rate, split=0)
        vald_dataset = H36M_Dataset(args.data_dir, args.input_n,
                            args.output_n, args.skip_rate, split=1)
        dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,   # 99 里面用的数？？
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])  # 共 66 个点 拆分看看

    elif args.loss_type == 'angle':
        dataset = H36M_Dataset_Angle(args.data_dir, args.input_n, args.output_n, 
                                    args.skip_rate, split=0)
        vald_dataset = H36M_Dataset_Angle(args.data_dir, args.input_n, 
                                    args.output_n, args.skip_rate, split=1)
        dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                        43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                        86])

    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,   # 这里的batch_size 是50
                            num_workers=args.num_worker, pin_memory=True, drop_last=True)
    vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_worker, pin_memory=True, drop_last=True)

    
    for epoch in range(args.n_epochs):
        print('Run epoch: %i'%epoch)
        running_loss = 0
        n = 0
        model.train()
        for cnt, batch in tqdm(enumerate(data_loader), total=len(data_loader)):  # 用来添加进度条的库
            batch = batch.to(device)
            batch_dim = batch.shape[0]
            n += batch_dim

            if args.loss_type == 'mpjpe':
                sequences_train = batch[:, 0:args.input_n, dim_used].view(
                    -1, args.input_n, args.pose_dim)
                sequences_gt = batch[:, args.input_n:args.input_n +  #  sequences_gt  表示目标数
                                    args.output_n, dim_used].view(-1, args.output_n, args.pose_dim)
            elif args.loss_type == 'angle':
                sequences_train=batch[:, 0:args.input_n, dim_used].view(  # input_n = 10
                      -1,args.input_n,len(dim_used))
                sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, dim_used]

            optimizer.zero_grad()

            if args.delta_x:  # 这里是True 预测两帧之间的差异
                sequences_all = torch.cat((sequences_train, sequences_gt), 1)  # 所有的数
                sequences_all_delta = [
                    sequences_all[:, 1, :] - sequences_all[:, 0, :]]  # 第1帧和第0帧的差
                for i in range(args.input_n+args.output_n-1):
                    sequences_all_delta.append(  # 每相邻的帧数的差值
                        sequences_all[:, i+1, :] - sequences_all[:, i, :])
    
                sequences_all_delta = torch.stack(
                    (sequences_all_delta)).permute(1, 0, 2)  # 交换维度 第一维度和第二维度交换
                sequences_train_delta = sequences_all_delta[:,  # sequences_all_delta.shape = [50, 35, 66]
                                                            0:args.input_n, :]
                #  sequences_train_delta.shape =  [50, 10, 66] ----batch = 50 | H = 22 | x\y\z = 3 | w = 10
                # print(sequences_train_delta.shape,"sequences_train_delta")
          
                sequences_predict = model(sequences_train_delta)  # 运用 model进行训练

                # sequences_predict.shape = [50, 25, 66]

                sequences_predict = delta_2_gt(  #  加上最后1帧的值
                    sequences_predict, sequences_train[:, -1, :])
                # print(sequences_predict.shape, "sequences_predict") # sequences_predict [50, 25, 66]
                loss = mpjpe_error(sequences_predict, sequences_gt)

                if args.use_relative_loss: # C.use_relative_loss = True
                    b, n, _ = sequences_predict.shape
                    sequences_predict_ = sequences_predict.reshape(b,n,22,3)
                    dsequences_predict = gen_velocity(sequences_predict_)
                    sequences_gt_ = sequences_gt.reshape(b,n,22,3)
                    dsequences_gt = gen_velocity(sequences_gt_)
                    dloss = mpjpe_error(dsequences_predict, dsequences_gt)
                    loss = loss + dloss
                else:
                    loss = loss.mean()
                
                

            elif args.loss_type == 'mpjpe':
                sequences_train = sequences_train/1000
                sequences_predict = model(sequences_train)
                loss = mpjpe_error(sequences_predict, sequences_gt)

            elif args.loss_type == 'angle':
                sequences_predict=model(sequences_train)              
                loss=torch.mean(torch.sum(torch.abs(sequences_predict.reshape(-1,args.output_n,len(dim_used)) - sequences_gt), dim=2).view(-1))


            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad)

            optimizer.step()

            running_loss += loss*batch_dim

        train_loss.append(running_loss.detach().cpu()/n)
        model.eval()
        with torch.no_grad():
            running_loss = 0
            n = 0
            for cnt, batch in enumerate(vald_loader):
                batch = batch.to(device)
                batch_dim = batch.shape[0]
                n += batch_dim

                if args.loss_type == 'mpjpe':
                    sequences_train = batch[:, 0:args.input_n, dim_used].view(
                        -1, args.input_n, args.pose_dim)  # pose_dim = 66 input_n = 10 number of model's input frames
                    sequences_gt = batch[:, args.input_n:args.input_n +
                                        args.output_n, dim_used].view(-1, args.output_n, args.pose_dim)
                elif args.loss_type == 'angle':
                    sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used))
                    sequences_gt=batch[:, args.input_n:args.input_n+args.output_n,:]


                if args.delta_x:
                    sequences_all = torch.cat(
                        (sequences_train, sequences_gt), 1)
                    sequences_all_delta = [
                        sequences_all[:, 1, :] - sequences_all[:, 0, :]]
                    for i in range(args.input_n+args.output_n-1):
                        sequences_all_delta.append(
                            sequences_all[:, i+1, :] - sequences_all[:, i, :])

                    sequences_all_delta = torch.stack(
                        (sequences_all_delta)).permute(1, 0, 2)
                    sequences_train_delta = sequences_all_delta[:,
                                                                0:args.input_n, :]  # 这里的10帧是帧数
                    sequences_predict = model(sequences_train_delta)
                    sequences_predict = delta_2_gt(
                        sequences_predict, sequences_train[:, -1, :])
                    loss = mpjpe_error(sequences_predict, sequences_gt)

                elif args.loss_type == 'mpjpe':
                    sequences_train = sequences_train/1000
                    sequences_predict = model(sequences_train)
                    loss = mpjpe_error(sequences_predict, sequences_gt)

                elif args.loss_type == 'angle':
                    all_joints_seq=batch.clone()[:, args.input_n:args.input_n+args.output_n,:]
                    sequences_predict=model(sequences_train)
                    all_joints_seq[:,:,dim_used] = sequences_predict
                    loss = euler_error(all_joints_seq,sequences_gt)

                running_loss += loss*batch_dim
            val_loss.append(running_loss.detach().cpu()/n)
        if args.use_scheduler:
            scheduler.step()

        if args.loss_type == 'mpjpe':
            test_loss.append(test_mpjpe(model, args))
        elif args.loss_type == 'angle':
            test_loss.append(test_angle(model, args))

        # tb_writer.add_scalar('loss/train', train_loss[-1].item(), epoch)
        # tb_writer.add_scalar('loss/val', val_loss[-1].item(), epoch)
        # tb_writer.add_scalar('loss/test', test_loss[-1].item(), epoch)

        torch.save(model.state_dict(), os.path.join(log_dir, 'model.pt'))
        # TODO write something to save the best model
        # if (epoch+1)%1==0:
        #     # print('----saving model-----')
        #     torch.save(model.state_dict(),os.path.join(args.model_path,model_name))
        if (epoch+1)%1==0:
            print('----saving model-----')
            model_name_ = 'hm36_3d_'+str(args.output_n)+'frames_ckpt_'+'epoch_'+str(epoch)
            # torch.save(model.state_dict(),os.path.join(args.model_path,model_name_))
            checkpoint = {
              "net": model.state_dict(),
              'optimizer':optimizer.state_dict(),
              "epoch": epoch
                    }
            torch.save(checkpoint,os.path.join(args.model_path,model_name_))


def test_mpjpe(model, args):
    t_3d_all = []
    t_3d_all_1 = []
    t_3d_all_2 = []
    t_3d_all_3 = []
    t_3d_all_4 = []
    t_3d_all_5 = []
    t_3d_all_6 = []
    t_3d_all_7 = []
    t_3d_all_8 = []
    N = 1
    eval_frame = [1, 3, 7, 9, 13, 17, 21, 24]
    t_3d = np.zeros(len(eval_frame))
    global save_loss
    device = args.dev
    model.eval()
    accum_loss = 0
   
    m_h36 = np.zeros([25])
    motion_h36m_target_length = 25
    results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']
    titles = np.array(range(motion_h36m_target_length)) + 1
    n_batches = 0  # number of batches for all the sequences
    actions = define_actions(args.actions_to_consider)
    if args.loss_type == 'mpjpe':
        dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                            26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                            46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                            75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    elif args.loss_type == 'angle':
        dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 
                            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 
                            56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85, 86])
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate(
        (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate(
        (joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    for action in actions:
        running_loss = 0
        n = 0
        if args.loss_type == 'mpjpe': # 是mpjpe类型
            dataset_test = H36M_Dataset(args.data_dir, args.input_n,
                                    args.output_n, args.skip_rate, split=2, actions=[action])
        elif args.loss_type == 'angle':
            dataset_test = H36M_Dataset_Angle(args.data_dir, args.input_n,
                                    args.output_n, args.skip_rate, split=2, actions=[action])
        # print('>>> Test dataset length: {:d}'.format(dataset_test.__len__()))

        test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test,
                                 shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
        for cnt, batch in enumerate(test_loader):
            with torch.no_grad():

                batch = batch.to(device)
                batch_dim = batch.shape[0]
                n += batch_dim

                all_joints_seq = batch.clone(
                )[:, args.input_n:args.input_n+args.output_n, :]
                all_joints_seq_gt = batch.clone(
                )[:, args.input_n:args.input_n+args.output_n, :]

                all_joints_seq_ = batch.clone(
                )[:, args.input_n:args.input_n+args.output_n, :]
                all_joints_seq_gt_ = batch.clone(
                )[:, args.input_n:args.input_n+args.output_n, :]

                sequences_train = batch[:, 0:args.input_n,
                                        dim_used].view(-1, args.input_n, len(dim_used))

                sequences_gt = batch[:, args.input_n:args.input_n +
                                     args.output_n, dim_used].view(-1, args.output_n, args.pose_dim)
                
                if args.delta_x:
                    sequences_all = torch.cat(
                        (sequences_train, sequences_gt), 1)
                    sequences_all_delta = [
                        sequences_all[:, 1, :] - sequences_all[:, 0, :]]
                    for i in range(args.input_n+args.output_n-1):
                        sequences_all_delta.append(
                            sequences_all[:, i+1, :] - sequences_all[:, i, :])

                    sequences_all_delta = torch.stack(
                        (sequences_all_delta)).permute(1, 0, 2)
                    sequences_train_delta = sequences_all_delta[:,
                                                                0:args.input_n, :]

                    sequences_predict = model(sequences_train_delta)

                    sequences_predict = delta_2_gt(
                        sequences_predict, sequences_train[:, -1, :])
                    loss = mpjpe_error(sequences_predict, sequences_gt)
                    N += n

                else:
                    sequences_train = sequences_train/1000
                    sequences_predict = model(sequences_train)
                    loss = mpjpe_error(sequences_predict, sequences_gt)

                all_joints_seq_[:, :, dim_used] = sequences_predict
                all_joints_seq_[:, :, index_to_ignore] = all_joints_seq[:, :, index_to_equal]

                all_joints_seq_gt_[:, :, dim_used] = sequences_gt
                all_joints_seq_gt_[:, :, index_to_ignore] = all_joints_seq_gt[:, :, index_to_equal]

                # all_joints_seq_gt.shape ([256, 25, 96])

                # 包涵32 全骨骼 不忽略的
                all_joints_seq_3d = all_joints_seq_.view(sequences_predict.shape[0], sequences_predict.shape[1], -1, 3)
                all_joints_seq_gt_3d = all_joints_seq_gt_.view(sequences_gt.shape[0], sequences_gt.shape[1], -1, 3)

                for k in np.arange(0, len(eval_frame)):
                    j = eval_frame[k]
                    # t_3d[k] += torch.mean(torch.norm(all_joints_seq_gt_3d[:, j, :, :].contiguous().view(-1, 3) - all_joints_seq_3d[:, j, :, :].contiguous().view(-1, 3), 2, 1)).item() * n
                    t_3d[k] += torch.sum(torch.mean(torch.norm(
                        all_joints_seq_gt_3d[:, j, :, :].contiguous() - all_joints_seq_3d[:, j, :, :].contiguous(),
                        dim=2), dim=1), dim=0).item()
                

            all_joints_seq[:, :, dim_used] = sequences_predict
            all_joints_seq[:, :,
                           index_to_ignore] = all_joints_seq[:, :, index_to_equal]

            all_joints_seq_gt[:, :, dim_used] = sequences_gt
            all_joints_seq_gt[:, :,
                              index_to_ignore] = all_joints_seq_gt[:, :, index_to_equal]

            loss = mpjpe_error(all_joints_seq.view(-1, args.output_n, 32, 3),
                               all_joints_seq_gt.view(-1, args.output_n, 32, 3))

            running_loss += loss*batch_dim
            accum_loss += loss*batch_dim

            all_joints_seq = all_joints_seq.view(-1, args.output_n, 32, 3)
            all_joints_seq_gt = all_joints_seq_gt.view(-1, args.output_n, 32, 3)

            mpjpe_h36 = torch.sum(torch.mean(torch.norm(all_joints_seq - all_joints_seq_gt, dim=3), dim=2),
                          dim=0)
            m_h36 += mpjpe_h36.cpu().numpy()

        n_batches += n

        t_3d_all_1.append(t_3d[0] / N)
        t_3d_all_2.append(t_3d[1] / N)
        t_3d_all_3.append(t_3d[2] / N)
        t_3d_all_4.append(t_3d[3] / N)
        t_3d_all_5.append(t_3d[4] / N)
        t_3d_all_6.append(t_3d[5] / N)
        t_3d_all_7.append(t_3d[6] / N)
        t_3d_all_8.append(t_3d[7] / N)

    print('overall average loss in mm is: %f'%(accum_loss/n_batches))
    
    print('overall 80mm loss in mm is: ', np.mean(t_3d_all_1))
    print('overall 160mm loss in mm is: ', np.mean(t_3d_all_2))
    print('overall 320mm loss in mm is: ', np.mean(t_3d_all_3))
    print('overall 400mm loss in mm is: ', np.mean(t_3d_all_4))
    print('overall 560mm loss in mm is: ', np.mean(t_3d_all_5))
    print('overall 720mm loss in mm is: ', np.mean(t_3d_all_6))
    print('overall 880mm loss in mm is: ', np.mean(t_3d_all_7))
    print('overall 1000mm loss in mm is: ', np.mean(t_3d_all_8))


    m_h36 = m_h36 / n_batches
    ret = {}
    for j in range(25):
      ret["#{:d}".format(titles[j])] = [m_h36[j], m_h36[j]]
    print("'#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25' loss is: \n", [round(ret[key][0], 1) for key in results_keys])
  

    if save_loss == 0:
      save_loss = accum_loss/n_batches
    
    else:
    
      if accum_loss/n_batches <= save_loss:
        save_loss = accum_loss/n_batches
        
        print('----saving best model-----')
        model_name_ = 'hm36_3d_'+str(args.output_n)+'frames_ckpt_'+'best'
        # torch.save(model.state_dict(),os.path.join(args.model_path,model_name_))
        checkpoint = {"net": model.state_dict(),}
        torch.save(checkpoint,os.path.join(args.model_path_best,model_name_))

      else:
        save_loss = save_loss

    #奔驰

    return m_h36
    # return accum_loss/n_batches


def test_angle(model, args):


    device = args.dev
    model.eval()
    accum_loss=0  
    n_batches=0 # number of batches for all the sequences
    actions=define_actions(args.actions_to_consider)
    dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                            43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                            86])

    for action in actions:
        running_loss=0
        n=0
        dataset_test = H36M_Dataset_Angle(args.data_dir,args.input_n,args.output_n,args.skip_rate, split=2,actions=[action])
        #print('>>> Test dataset length: {:d}'.format(dataset_test.__len__()))

        test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
        for cnt, batch in enumerate(test_loader):
            with torch.no_grad():

                batch=batch.to(device)
                batch_dim=batch.shape[0]
                n+=batch_dim
                
                all_joints_seq=batch.clone()[:, args.input_n:args.input_n+args.output_n,:]

                sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used))
                sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, :]

                sequences_predict=model(sequences_train)
                all_joints_seq[:,:,dim_used] = sequences_predict
                loss=euler_error(all_joints_seq,sequences_gt)

                running_loss+=loss*batch_dim
                accum_loss+=loss*batch_dim

        n_batches+=n
    print('overall average loss in euler angle is: '+str(accum_loss/n_batches))
    
    return accum_loss/n_batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False) # Parameters for mpjpe
    parser.add_argument('--data_dir', type=str, default=r'/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/h36m/data_h36m', help='path to the unziped dataset directories(H36m/AMASS/3DPW)')
    parser.add_argument('--input_n', type=int, default=10, help="number of model's input frames")
    parser.add_argument('--output_n', type=int, default=25, help="number of model's output frames")
    parser.add_argument('--skip_rate', type=int, default=1, choices=[1, 5], help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers in the dataloader')
    parser.add_argument('--root', default=r'/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/h36m/run_for_longging', type=str, help='root path for the logging') #'./runs'

    parser.add_argument('--activation', default='mish', type=str, required=False) 
    parser.add_argument('--r_se', default=8, type=int, required=False)

    parser.add_argument('--n_epochs', default=25, type=int, required=False)
    parser.add_argument('--batch_size', default=50, type=int, required=False)  
    parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
    parser.add_argument('--pin_memory', default=False, type=bool, required=False)
    parser.add_argument('--loader_workers', default=4, type=int, required=False)
    parser.add_argument('--load_checkpoint', default=False, type=bool, required=False)
    parser.add_argument('--dev', default='cuda:0', type=str, required=False)
    parser.add_argument('--initialization', type=str, default='none', help='none, glorot_normal, glorot_uniform, hee_normal, hee_uniform')
    parser.add_argument('--use_scheduler', default=True, type=bool, required=False)
    parser.add_argument('--milestones', type=list, default=[5, 10], help='[5, 10, 20, 30, 40, 50, 60, 70, 80, 90]the epochs after which the learning rate is adjusted by gamma')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma correction to the learning rate, after reaching the milestone epochs')
    parser.add_argument('--clip_grad', type=float, default=None, help='select max norm to clip gradients')
    parser.add_argument('--model_path', type=str, default='/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/checkpoints/hm36', help='directory with the models checkpoints ')
    parser.add_argument('--actions_to_consider', default='all', help='Actions to visualize.Choose either all or a list of actions')
    parser.add_argument('--batch_size_test', type=int, default=256, help='batch size for the test set')
    parser.add_argument('--visualize_from', type=str, default='test', choices=['train', 'val', 'test'], help='choose data split to visualize from(train-val-test)')
    parser.add_argument('--loss_type', type=str, default='mpjpe', choices=['mpjpe', 'angle'])
    parser.add_argument('--use_relative_loss', type=bool, default=True)
    parser.add_argument('--model_path_best', type=str, default='/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/checkpoints', help='directory with the best models checkpoints ')

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
        parser_angle.add_argument('--hidden_dim', default=10, type=int, required=False) 
        parser_angle.add_argument('--num_blocks', default=3, type=int, required=False) 
        parser_angle.add_argument('--tokens_mlp_dim', default=40, type=int, required=False)
        parser_angle.add_argument('--channels_mlp_dim', default=60, type=int, required=False) 
        parser_angle.add_argument('--regularization', default=0.0, type=float, required=False)
        parser_angle.add_argument('--pose_dim', default=48, type=int, required=False)
        parser_angle.add_argument('--lr', default=1e-02, type=float, required=False) 
        args = parser_angle.parse_args()
    
    if args.loss_type == 'angle' and args.delta_x:
        raise ValueError('Delta_x and loss type angle cant be used together.')

    # print(args)
    # model 的位置                     # 66                     # 3
    model = MlpMixer(num_classes=args.pose_dim, num_blocks=args.num_blocks,  # args.pose_dim = 66 args.num_blocks = 4
                     hidden_dim=args.hidden_dim, tokens_mlp_dim=args.tokens_mlp_dim, # hidden_dim = 50   tokens_mlp_dim = 20
                     channels_mlp_dim=args.channels_mlp_dim, seq_len=args.input_n, # channels_mlp_dim = 50  input_n = 10
                     pred_len=args.output_n, activation=args.activation,  # output_n = 25   activation = mish
                     mlp_block_type='normal', regularization=args.regularization, # regularization = 0.1
                     input_size=args.pose_dim, initialization='none', r_se=args.r_se,  # pose_dim = 66 r_se = 8
                     use_max_pooling=False, use_se=True)  # 使用se block 模块



    model = model.to(args.dev)

    print('total number of parameters of the network is: ' +
          str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    model_name = 'h36_3d_'+str(args.output_n)+'frames_ckpt'

    train(model, model_name, args)
    test_mpjpe(model, args)
