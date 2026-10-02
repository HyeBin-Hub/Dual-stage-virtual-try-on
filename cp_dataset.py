################################################################################################
###################################### Dress_Code code #########################################
################################################################################################
#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json

import os
from model_config import validate_resolution

class CPDataset(data.Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.stage = opt.stage # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        validate_resolution(self.fine_height, self.fine_width)
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_gray = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "CPDataset"

    def _load_image(self, folder, name, mode):
        path = osp.join(self.data_path, folder, name)
        with Image.open(path) as image:
            if image.size != (self.fine_width, self.fine_height):
                raise ValueError(
                    f'Wrong image size: {path}: {image.size}, '
                    f'expected {(self.fine_width, self.fine_height)} (width, height). '
                    'Use consistently preprocessed images and matching resolution options.'
                )
            return image.convert(mode)
    
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # cloth image & cloth mask
        if self.stage == 'GMM':
            c = self._load_image('cloth_deformation', c_name, 'RGB')
            cm = self._load_image('cloth_mask_deformation', c_name, 'L')
            
            #####################################################################################
            #                          Load GT Person Upper Cloth image 
            #####################################################################################
            # gt_cltoh_warp_mask = Image.open(osp.join(self.data_path, 'cloth_gt', c_name)).convert('L')
            
            # # (gt_cloth_warp_mask : Tensor) 
            # gt_cltoh_warp_mask_array = np.array(gt_cltoh_warp_mask)
            # gt_cltoh_warp_mask_array = (gt_cltoh_warp_mask_array >= 128).astype(np.float32)
            # gt_cltoh_warp_mask = torch.from_numpy(gt_cltoh_warp_mask_array) # [0,1]
            # gt_cltoh_warp_mask.unsqueeze_(0)
            #####################################################################################
            
            
            

        else:
            warp_dir = getattr(self.opt, 'warp_cloth_dir', 'paired_warp_cloth-2')
            warp_name = im_name if getattr(self.opt, 'warp_cloth_key', 'person') == 'person' else c_name
            c = self._load_image(warp_dir, warp_name, 'RGB')
            # cm = Image.open(osp.join(self.data_path, 'warp-mask', c_name)).convert('L')
            
            #             if self.datamode == 'train':
            #####################################################################################
            #                          Load GT Person Upper Cloth image 
            #####################################################################################
            # gt_cltoh_warp_mask = Image.open(osp.join(self.data_path, 'cloth_gt', c_name)).convert('L')
            
            # # (gt_cloth_warp_mask : Tensor) 
            # gt_cltoh_warp_mask_array = np.array(gt_cltoh_warp_mask)
            # gt_cltoh_warp_mask_array = (gt_cltoh_warp_mask_array >= 128).astype(np.float32)
            # gt_cltoh_warp_mask = torch.from_numpy(gt_cltoh_warp_mask_array) # [0,1]
            # gt_cltoh_warp_mask.unsqueeze_(0)
            #####################################################################################
     
        c = self.transform(c)  # [-1,1]
        
        if self.stage == 'GMM':
            cm = torch.from_numpy((np.array(cm) >= 128).astype(np.float32)).unsqueeze(0)

        if getattr(self.opt, 'require_gt_mask', False):
            mask_dir = getattr(self.opt, 'gt_mask_dir', 'cloth_gt')
            mask_path = osp.join(self.data_path, mask_dir, c_name)
            if not osp.isfile(mask_path):
                raise FileNotFoundError('Training requires a garment ground-truth mask: ' + mask_path)
            mask_img = self._load_image(mask_dir, c_name, 'L')
            gt_cltoh_warp_mask = torch.from_numpy(
                (np.array(mask_img) >= 128).astype(np.float32)
            ).unsqueeze(0)

        # person image 
        im = self._load_image('images', im_name, 'RGB')
        im = self.transform(im) # [-1,1]

        # load parsing image
        parse_name = im_name .replace('.jpg', '.png')
        im_parse = self._load_image('image_parse', parse_name, 'L')
        parse_array = np.array(im_parse)
        
        parse_shape = (parse_array > 0).astype(np.float32)
    
        parse_head = (parse_array > 20).astype(np.float32)

        parse_cloth = (parse_array == 15).astype(np.float32) 
        
        ########################################################################################################
        # ########################################################################################################
        # # Save parse_shape
        # parse_shape_save_path = "./result/parse_shape"
        # if not os.path.exists(parse_shape_save_path):
        #     os.makedirs(parse_shape_save_path, exist_ok=True)
        # Image.fromarray((parse_shape * 255).astype(np.uint8)).save(osp.join(parse_shape_save_path, im_name))

        # # Save parse_head
        # parse_head_save_path = "./result/parse_head"
        # if not os.path.exists(parse_head_save_path):
        #     os.makedirs(parse_head_save_path, exist_ok=True)
        # Image.fromarray((parse_head * 255).astype(np.uint8)).save(osp.join(parse_head_save_path, im_name))

        # # Save parse_cloth
        # parse_cloth_save_path = "./result/parse_cloth"
        # if not os.path.exists(parse_cloth_save_path):
        #     os.makedirs(parse_cloth_save_path, exist_ok=True)
        # Image.fromarray((parse_cloth * 255).astype(np.uint8)).save(osp.join(parse_cloth_save_path, im_name))
        # ########################################################################################################
        ########################################################################################################

       
        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        shape = self.transform_gray(parse_shape) # [-1,1]
    
        phead = torch.from_numpy(parse_head) # [0,1]

        pcm = torch.from_numpy(parse_cloth) # [0,1]
        
        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        im_h = im * phead  - (1 - phead) # [-1,1], fill 0 for other parts
        
        #####################################################################################
        #                        Load Person Cloth Agnostic-v3.2 image
        #####################################################################################
        agnostics_img = self._load_image('agnostic-v3.2', im_name, 'RGB')
    
        # agnostics (agnostics : Tensor)
        ag = self.transform(agnostics_img)
        #####################################################################################
        
        ####################################################################################
        #                                Load Keypoint image
        #####################################################################################
        im_names = im_name.split(".")[0] + ".jpg" 
        keypoint_img = self._load_image('skeletons', im_names, 'RGB')
        
        # Keypoint (keypoint_img : Tensor)
        kp = self.transform(keypoint_img) # [-1,1]
        #####################################################################################
        
        # load pose points
        # pose_name = im_name.replace('.jpg', '_keypoints.json')
        pose_name = im_name.replace('.jpg', '.json')
        with open(osp.join(self.data_path, 'keypoints', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 4))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            # pointx = pose_data[i,0]
            # pointy = pose_data[i,1]
            # pointx, pointy, confidence = pose_data[i][:3]
            pointx = np.multiply(pose_data[i, 0], self.fine_width / 384.0)
            pointy = np.multiply(pose_data[i, 1], self.fine_height / 512.0)
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform_gray(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform_gray(im_pose)
        
        # cloth-agnostic representation
        # agnostic = torch.cat([shape, im_h, pose_map], 0) 
        agnostic = torch.cat([ag, kp], 0)

        if self.stage == 'GMM':
            im_g = Image.open(osp.join(osp.dirname(__file__), 'grid.png')).convert('RGB')
            im_g = im_g.resize((self.fine_width, self.fine_height), Image.BILINEAR)
            im_g = self.transform(im_g)
        else:
            im_g = ''

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            # 'cloth_mask':     cm,   # for input
            'image':    im,         # for visualization
            'agnostic': agnostic,   # for input
            'parse_cloth': im_c,    # for ground truth
            'shape': shape,         # for visualization
            'head': im_h,           # for visualization
            'pose_image': im_pose,  # for visualization
            'grid_image': im_g,     # for visualization
            # 'gt_cltoh_warp_mask':gt_cltoh_warp_mask,
            }

        if self.stage == 'GMM':
            result['cloth_mask'] = cm
        if getattr(self.opt, 'require_gt_mask', False):
            result['gt_cltoh_warp_mask'] = gt_cltoh_warp_mask
        return result

    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=False,
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
