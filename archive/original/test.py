#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from GMM_networks import GMM, load_checkpoint

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images


# from TOM_Fusion_AND_GT_U_Net import Fusion_and_GT_UNet


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "Dresscode-256")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "test_pairs_unpaired_modified.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result-dress-256', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    
    parser.add_argument('--BASE', type=str, default='test_paired_try_on', help='model checkpoint for test')
    
    parser.add_argument("--display_count", type=int, default = 1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.BASE)
    
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    def make_dir(dir_name):
        dir_path = os.path.join(save_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path
        
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        c_names = inputs['c_name']
        im_name = inputs['im_name']
        
        im = inputs['image'].cuda()
        # im_pose = inputs['pose_image'].cuda()
        shape = inputs['shape'].cuda()
        
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        
        cm = inputs['cloth_mask'].cuda()
        # ce = inputs['cloth_edge'].cuda()
        
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        
        # im_h_l = inputs['im_h_l'].cuda()
            
        grid, theta, one_warped_cloth, output = model(agnostic, c)
        
        one_warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        one_warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        one_warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        # one_warped_edge = F.grid_sample(ce, grid, padding_mode='zeros')

        c_rendered, m_composite = torch.split(output, 3,1)
        c_rendered = F.tanh(c_rendered)
        m_composite = F.sigmoid(m_composite)
        c_result = one_warped_cloth * m_composite + c_rendered * (1 - m_composite)
    
        # visuals = [ [im_h_l, shape, im_pose], 
        #            [c, c_result, im_c], 
        #            [m_composite, (c_result+im)*0.5, c_result]]
        
        # warped_cloth_dir = make_dir('warped_cloth')
        # save_images(one_warped_cloth, c_names, warped_cloth_dir) 
        
        # warped_mask_dir = make_dir('warped_mask')
        # save_images(one_warped_mask*2-1, c_names, warped_mask_dir) 
        
        c_result_dir = make_dir('c_result')
        save_images(c_result, c_names, c_result_dir) 

        if (step+1) % opt.display_count == 0:
            # board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)
        


def test_tom(opt, test_loader, model, board):
    model.cuda()
    model.eval()
    
    base_name = os.path.basename(opt.BASE)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        # cm = inputs['cloth_mask'].cuda()
        
        # real_c = inputs['real_c'].cuda()
        
        outputs = model(torch.cat([agnostic, c],1))
        
        # input_img = torch.cat([agnostic, c],1)
        # outputs = model(input_img, real_c)
        
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        # visuals = [ [im_h, shape, im_pose], 
        #            [c, 2*cm-1, m_composite], 
        #            [p_rendered, p_tryon, im]]
            
        save_images(p_tryon, im_names, try_on_dir) 
        if (step+1) % opt.display_count == 0:
            # board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)

# from TOM_network_Fusion_u_net import FusionUNet
# from TOM_network_GT_U_Net import GT_U_Net
# from TOM_network_Trans_block_U_net_transformer_GT_U_NET import U_Transformer

# from Ablation_TOM_GT_Fusion import FusionUNet
# from Ablation_TOM_GT_Cross import U_Transformer

from TOM_Fusion_GT_U_Transformer import FusionUNet

def main():
    opt = get_opt()
    print(opt)
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, train_loader, model, board)
    elif opt.stage == 'TOM':
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        
        # model = FusionUNet(9, 4, 64)
       
        # model = Fusion_and_GT_UNet(9, 4, 64)
        
        # model = GT_U_Net(9,4)
        
        # model = U_Transformer(9, 4)
        
        # model = FusionUNet(9, 4, 64)
        
        model = FusionUNet(9, 4, 64)
        
        load_checkpoint(model, opt.checkpoint)
        
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
  
    print('Finished test %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
