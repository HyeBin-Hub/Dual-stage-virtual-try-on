#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader

from GMM_networks import GMM, load_checkpoint, save_checkpoint
from model_config import validate_resolution

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images,save_images

from loss import GicLoss, VGGLoss


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "Dresscode-256")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM", choices=['GMM', 'TOM'])
    parser.add_argument('--warp_cloth_dir', default='paired_warp_cloth-2')
    parser.add_argument('--warp_cloth_key', choices=['person', 'garment'], default='person')
    parser.add_argument('--gt_mask_dir', default='cloth_gt')
    parser.add_argument("--data_list", default = "train_pairs_modified.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints-dresscode-256', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

###############################################################################################################
# ------------------------------------------------- GMM model -------------------------------------------------
###############################################################################################################

def train_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterion_L1 = nn.L1Loss()
    criterion_VGG = VGGLoss()
    criterion_shape = nn.L1Loss()
    gicloss = GicLoss(opt)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        c_name = inputs['c_name']
        im = inputs['image'].cuda()
        # im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        # im_h_l = inputs['im_h_l'].cuda()
        gt_cltoh_warp_mask = inputs['gt_cltoh_warp_mask'].cuda()
        # ce = inputs['cloth_edge'].cuda()
            
        grid, theta, one_warped_cloth, output= model(agnostic, c)
        
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
        
        save_images(im_c, c_name, "im_c")
        
        loss_one_cloth = criterion_L1(one_warped_cloth, im_c) + 0.1 *  criterion_VGG(one_warped_cloth, im_c)
        loss_two_cloth = criterion_L1(one_warped_cloth, im_c) + 0.1 *  criterion_VGG(c_result, im_c)
        
        loss_one_cloth_mask = criterion_shape(one_warped_mask, gt_cltoh_warp_mask) 
        loss_two_cloth_mask = criterion_shape(m_composite, gt_cltoh_warp_mask) 
        
        # loss_one_cloth_edge = criterion_L1(one_warped_edge, edge_gt_cltoh_warp) 
        
        # loss_style = criterion_style(c_result, im_c) 
        
        # loss_tv = TVLoss(c_result)
        
        Lgic = gicloss(grid)
        Lgic = Lgic / (grid.shape[0] * grid.shape[1] * grid.shape[2])
        
        loss = (loss_one_cloth + loss_two_cloth)  + \
                (loss_one_cloth_mask * 0.1 + loss_two_cloth_mask * 0.1) + \
                40 * Lgic  # + loss_style * 0.001 #  + loss_tv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            # board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))
     

###############################################################################################################
# ------------------------------------------------- TOM model -------------------------------------------------
###############################################################################################################


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    # criterionTV = TotalVariation()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        # cm = inputs['cloth_mask'].cuda()
        gt_cltoh_warp_mask = inputs['gt_cltoh_warp_mask'].cuda()
        # real_c = inputs['real_c'].cuda()
        
        # input_img = torch.cat([agnostic, c],1)
        # outputs = model(input_img, real_c)
        
        outputs = model(torch.cat([agnostic, c],1))
        
        p_rendered, m_composite = torch.split(outputs, 3,1)
        
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, gt_cltoh_warp_mask*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
            
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, gt_cltoh_warp_mask)
        
        # loss_tv = criterionTV(p_tryon)
        
        loss = loss_l1 * 0.1 + loss_vgg * 0.3 + loss_mask # + loss_tv
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f' 
                    % (step+1, t, loss.item(), loss_l1.item(), 
                    loss_vgg.item(), loss_mask.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))

###############################################################################################################
# ------------------------------------------------- Main -------------------------------------------------
###############################################################################################################

from TOM_Fusion_GT_U_Transformer import FusionUNet


def main():
    opt = get_opt()
    opt.require_gt_mask = True
    validate_resolution(opt.fine_height, opt.fine_width)
    if not torch.cuda.is_available():
        raise RuntimeError('The supplied research models require a CUDA GPU.')
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if opt.checkpoint:
            load_checkpoint(model, opt.checkpoint)
            
        train_gmm(opt, train_loader, model, board)
        
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
        
    elif opt.stage == 'TOM':
        
        
        
        
        
        model = FusionUNet(9, 4, 64)
        
        if opt.checkpoint:
            load_checkpoint(model, opt.checkpoint)
            
        train_tom(opt, train_loader, model, board)
        
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
        
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
        
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
