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
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
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
    
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # cloth image & cloth mask
        if self.stage == 'GMM':
            c = Image.open(osp.join(self.data_path, 'cloth_deformation', c_name)).convert('RGB')
            cm = Image.open(osp.join(self.data_path, 'cloth_mask_deformation', c_name)).convert('L')
            
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
            c = Image.open(osp.join(self.data_path, 'paired_warp_cloth-2', c_name))
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
        
        # cm_array = np.array(cm)
        # cm_array = (cm_array >= 128).astype(np.float32)
        # cm = torch.from_numpy(cm_array) # [0,1]
        # cm.unsqueeze_(0)

        # person image 
        im = Image.open(osp.join(self.data_path, 'images', im_name)).convert("RGB")
        im = self.transform(im) # [-1,1]

        # load parsing image
        parse_name = im_name .replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image_parse', parse_name)).convert('L')
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
        shape = self.transform(parse_shape) # [-1,1]
    
        phead = torch.from_numpy(parse_head) # [0,1]

        pcm = torch.from_numpy(parse_cloth) # [0,1]
        
        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        im_h = im * phead  - (1 - phead) # [-1,1], fill 0 for other parts
        
        #####################################################################################
        #                        Load Person Cloth Agnostic-v3.2 image
        #####################################################################################
        agnostics_img = Image.open(osp.join(self.data_path, 'agnostic-v3.2', im_name))
    
        # agnostics (agnostics : Tensor)
        ag = self.transform(agnostics_img)
        #####################################################################################
        
        ####################################################################################
        #                                Load Keypoint image
        #####################################################################################
        im_names = im_name.split(".")[0] + ".jpg" 
        keypoint_img = Image.open(osp.join(self.data_path, 'skeletons', im_names))
        
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
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform(im_pose)
        
        # cloth-agnostic representation
        # agnostic = torch.cat([shape, im_h, pose_map], 0) 
        agnostic = torch.cat([ag, kp], 0)

        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
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
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
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


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    
    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()
    
    from IPython import embed; embed()



# #coding=utf-8
# import torch
# import torch.utils.data as data
# import torchvision.transforms as transforms

# from PIL import Image
# from PIL import ImageDraw

# import os.path as osp
# import numpy as np
# import json

# class CPDataset(data.Dataset):
#     """Dataset for CP-VTON.
#     """
#     def __init__(self, opt):
#         super(CPDataset, self).__init__()
#         # base setting
#         self.opt = opt
#         self.root = opt.dataroot
#         self.datamode = opt.datamode # train or test or self-defined
#         self.stage = opt.stage # GMM or TOM
#         self.data_list = opt.data_list
#         self.fine_height = opt.fine_height
#         self.fine_width = opt.fine_width
#         self.radius = opt.radius
#         self.data_path = osp.join(opt.dataroot, opt.datamode)
#         self.transform = transforms.Compose([  \
#                 transforms.ToTensor(),   \
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
#         # load data list
#         im_names = []
#         c_names = []
#         with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
#             for line in f.readlines():
#                 im_name, c_name = line.strip().split()
#                 im_names.append(im_name)
#                 c_names.append(c_name)

#         self.im_names = im_names
#         self.c_names = c_names

#     def name(self):
#         return "CPDataset"

#     def __getitem__(self, index):
#         c_name = self.c_names[index]
#         im_name = self.im_names[index]

#         # cloth image & cloth mask
#         if self.stage == 'GMM':
#             #####################################################################################
#             #                                    Load Cloth image
#             #####################################################################################
#             # Load Cloth
#             c = Image.open(osp.join(self.data_path, 'resize_cloth_deformation', c_name))
#             # Load Cloth Mask
#             cm = Image.open(osp.join(self.data_path, 'resize_cloth_mask_deformation', c_name))
#             # Load Cloth Edege
#             # ce = Image.open(osp.join(self.data_path, 'resize_edge_cloth_deformation', c_name))
#             #####################################################################################
            
#             if self.datamode == 'train':
#                 #####################################################################################
#                 #                          Load GT Person Upper Cloth image 
#                 #####################################################################################
#                 gt_cltoh_warp_mask = Image.open(osp.join(self.data_path, 'resize_gt_cloth_warped_mask', c_name))
                
#                 # (gt_cloth_warp_mask : Tensor) 
#                 gt_cltoh_warp_mask_array = np.array(gt_cltoh_warp_mask)
#                 gt_cltoh_warp_mask_array = (gt_cltoh_warp_mask_array >= 128).astype(np.float32)
#                 gt_cltoh_warp_mask = torch.from_numpy(gt_cltoh_warp_mask_array) # [0,1]
#                 gt_cltoh_warp_mask.unsqueeze_(0)
#                 #####################################################################################
                
#                 #####################################################################################
#                 #                       Load GT Person Upper Edge Cloth image 
#                 #####################################################################################
#                 # edge_gt_cltoh_warp = Image.open(osp.join(self.data_path, 'resize_edge_im_c', c_name))
            
#                 # # (edge_gt_cltoh_warp_array : Tensor) 
#                 # edge_gt_cltoh_warp_array = np.array(edge_gt_cltoh_warp)
#                 # edge_gt_cltoh_warp_array = (edge_gt_cltoh_warp_array >= 128).astype(np.float32)
#                 # edge_gt_cltoh_warp = torch.from_numpy(edge_gt_cltoh_warp_array) # [0,1]
#                 # edge_gt_cltoh_warp.unsqueeze_(0)
#                 #####################################################################################
                
            
#         else:
#             #####################################################################################
#             #                               Load Warped Cloth image 
#             #####################################################################################
#             # Load Warped Cloth 
            
#             ## test unpaird
#             # c = Image.open(osp.join(self.data_path, 'test_unpairs', c_name))
#             ## test paird
#             # c = Image.open(osp.join(self.data_path, 'test_pairs', im_name))
#             ## train paired
#             c = Image.open(osp.join(self.data_path, 'warp-cloth', c_name))
            
            
#             # Load Warped Cloth Mask
#             # cm = Image.open(osp.join(self.data_path, 'warp-mask', c_name))
#             # Load Warped Cloth Edge
#             # ce = Image.open(osp.join(self.data_path, 'warp-edge', c_name))
            
#             real_c = Image.open(osp.join(self.data_path, 'resize_cloth_deformation', c_name))
#             #####################################################################################
             
#             if self.datamode == 'train':
#                 #####################################################################################
#                 #                          Load GT Person Upper Cloth image 
#                 #####################################################################################
#                 gt_cltoh_warp_mask = Image.open(osp.join(self.data_path, 'resize_gt_cloth_warped_mask', c_name))
                
#                 # (gt_cloth_warp_mask : Tensor) 
#                 gt_cltoh_warp_mask_array = np.array(gt_cltoh_warp_mask)
#                 gt_cltoh_warp_mask_array = (gt_cltoh_warp_mask_array >= 128).astype(np.float32)
#                 gt_cltoh_warp_mask = torch.from_numpy(gt_cltoh_warp_mask_array) # [0,1]
#                 gt_cltoh_warp_mask.unsqueeze_(0)
#                 #####################################################################################
           
                
     
#         #####################################################################################
#         #                                Load (Waroed) Cloth image
#         #####################################################################################
#         # Cloth (c : Tensor)
#         c = self.transform(c)  # [-1,1]
        
#         # real_c = self.transform(real_c)
        
#         # Cloth mask (cm : Tensor)
#         cm_array = np.array(cm)
#         cm_array = (cm_array >= 128).astype(np.float32)
#         cm = torch.from_numpy(cm_array) # [0,1]
#         cm.unsqueeze_(0)
        
#         # # Cloth Edge (ce : Tensor)
#         # ce_array = np.array(ce)
#         # ce_array = (ce_array >= 128).astype(np.float32)
#         # ce = torch.from_numpy(ce_array) # [0,1]
#         # ce.unsqueeze_(0)
#         #####################################################################################

#         #####################################################################################
#         #                                Load Keypoint image
#         #####################################################################################
#         im_names = im_name.split(".")[0] + "_rendered.jpg" 
#         keypoint_img = Image.open(osp.join(self.data_path, 'resize_openpose_img', im_names))
        
#         # Keypoint (keypoint_img : Tensor)
#         kp = self.transform(keypoint_img) # [-1,1]
#         #####################################################################################
        
#         #####################################################################################
#         #                        Load Person Cloth Agnostic-v3.2 image
#         #####################################################################################
#         agnostics_img = Image.open(osp.join(self.data_path, 'resize_agnostic-v3.2', im_name))
        
#         # agnostics (agnostics : Tensor)
#         ag = self.transform(agnostics_img)
#         #####################################################################################
        
#         #####################################################################################
#         #            Load Person Segmentation(Parsing) Cloth Agnostic-v3.2 image
#         ##################################################################################### 
#         # parse_agnostic_img = Image.open(osp.join(self.data_path, 'resize_image-parse-agnostic-v3.2', im_name))
        
#         # # parse_agnostic (parse_agnostic : Tensor)
#         # pag = self.transform(parse_agnostic_img)
#         #####################################################################################
        
#         #####################################################################################
#         #                             Load DensePose image
#         #####################################################################################        
#         # dense_name = im_name.replace('.jpg', '.png')
#         # dense_img = Image.open(osp.join(self.data_path, 'resize_dense', dense_name))
        
#         # # Dense (dense : Tensor)
#         # dense = self.transform(dense_img)
        
#         # dense_array = np.array(dense)
        
#         # mask_clothes = (dense_array == 4).astype(np.float32)
#         # mask_clothes = self.transform(mask_clothes)
        
#         # mask_hair = (dense_array == 1).astype(np.float32)
#         # mask_hair = self.transform(mask_hair)
        
#         # mask_bottom = (dense_array == 8).astype(np.float32)
#         # mask_bottom = self.transform(mask_bottom)
        
#         # mask_head = (dense_array == 12).astype(np.float32)
#         # mask_head = self.transform(mask_head)
        
#         # mask_fore = (dense_array > 0).astype(np.float32)
#         # mask_fore = self.transform(mask_fore)
#         #####################################################################################

#         #####################################################################################
#         #                                Load Person image
#         ##################################################################################### 
#         im = Image.open(osp.join(self.data_path, 'resize_image', im_name))
        
#         # Persong (im : Tensor)
#         im = self.transform(im) # [-1,1]
#         #####################################################################################


#         #####################################################################################
#         #                       Load Person Segmentation(Parsing) image
#         ##################################################################################### 
#         parse_name = im_name.replace('.jpg', '.png')
#         im_parse = Image.open(osp.join(self.data_path, 'resize_image-parse-v3', parse_name))
        
#         parse_array = np.array(im_parse)
        
#         parse_shape = (parse_array > 0).astype(np.float32)
        
#         parse_head = (parse_array == 1).astype(np.float32) + \
#                      (parse_array == 2).astype(np.float32) + \
#                      (parse_array == 4).astype(np.float32) + \
#                      (parse_array == 13).astype(np.float32)
                
#         parse_agnostics = (parse_array == 1).astype(np.float32) + \
#                           (parse_array == 2).astype(np.float32) + \
#                           (parse_array == 4).astype(np.float32) + \
#                           (parse_array == 9).astype(np.float32) + \
#                           (parse_array == 12).astype(np.float32) + \
#                           (parse_array == 13).astype(np.float32) + \
#                           (parse_array == 16).astype(np.float32) + \
#                           (parse_array == 17).astype(np.float32)  # CP-VTON+ TOM input (reserved regions)
        
#         parse_cloth = (parse_array == 5).astype(np.float32) + \
#                       (parse_array == 6).astype(np.float32) + \
#                       (parse_array == 7).astype(np.float32)
                      
#         parse_agnostic_arm = (parse_array == 1).astype(np.float32) + \
#                              (parse_array == 2).astype(np.float32) + \
#                              (parse_array == 4).astype(np.float32) + \
#                              (parse_array == 9).astype(np.float32) + \
#                              (parse_array == 12).astype(np.float32) + \
#                                 (parse_array == 14).astype(np.float32) + \
#                                 (parse_array == 15).astype(np.float32) + \
#                              (parse_array == 13).astype(np.float32) + \
#                              (parse_array == 16).astype(np.float32) + \
#                              (parse_array == 17).astype(np.float32) 
       
#         # shape downsample
#         parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
#         parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
#         parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        
#         # Silhouette (shape : Tensor)
#         shape = self.transform(parse_shape) # [-1,1]
        
#         # parse_head (phead : Tensor)
#         phead = torch.from_numpy(parse_head) # [0,1]
#         # parse_cloth (pcm : Tensor)
#         pcm = torch.from_numpy(parse_cloth) # [0,1]
#         # parse_agnostics (pagnostic : Tensor)
#         pagnostic = torch.from_numpy(parse_agnostics)
        
#         # upper cloth
#         im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
#         im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts
#         im_h_l = im * pagnostic + (1 - pagnostic)
#         #####################################################################################

#         #####################################################################################
#         #                               Load Pose Points 
#         ##################################################################################### 
#         pose_name = im_name.replace('.jpg', '_keypoints.json')
#         with open(osp.join(self.data_path, 'resize_openpose_json', pose_name), 'r') as f:
#             pose_label = json.load(f)
#             pose_data = pose_label['people'][0]['pose_keypoints_2d']
#             pose_data = np.array(pose_data)
#             pose_data = pose_data.reshape((-1,3))

#         point_num = pose_data.shape[0]
#         pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
#         r = self.radius
#         im_pose = Image.new('L', (self.fine_width, self.fine_height))
#         pose_draw = ImageDraw.Draw(im_pose)
#         for i in range(point_num):
#             one_map = Image.new('L', (self.fine_width, self.fine_height))
#             draw = ImageDraw.Draw(one_map)
#             pointx = pose_data[i,0]
#             pointy = pose_data[i,1]
#             if pointx > 1 and pointy > 1:
#                 draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
#                 pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
#             one_map = self.transform(one_map)
#             pose_map[i] = one_map[0]

#         # just for visualization 
#         # (im_pose : Tensor)
#         im_pose = self.transform(im_pose)
#         #####################################################################################
        
        
#         #####################################################################################
#         #                           Cloth-Agnostic Representation
#         ##################################################################################### 
#         # # My Proposed Person Representation 
#         # agnostic = torch.cat([ag, kp], 0)
        
#         # GGAM_Ablation_1
#         agnostic = torch.cat([shape, im_h, pose_map], 0)
        
#         #####################################################################################


#         #####################################################################################
#         #                                  Load Grid image
#         #####################################################################################
#         if self.stage == 'GMM':
#             im_g = Image.open('grid.png')
#             im_g = self.transform(im_g)
#         else:
#             im_g = ''
#         #####################################################################################
        
#         #####################################################################################
#         #                                  
#         #####################################################################################
#         if self.stage == 'GMM':
#             if self.datamode == 'train':
#                 result = {
#                     'c_name':   c_name,     # for visualization
#                     'im_name':  im_name,    # for visualization or ground truth
                
#                     'cloth':    c,          # for input
#                     'cloth_mask':     cm,   # for input
                
#                     'image':    im,         # for visualization
                
#                     'agnostic': agnostic,   # for input
                
#                     'parse_cloth': im_c,    # for ground truth
#                     'shape': shape,         # for visualization
#                     'head': im_h,           # for visualization
#                     # 'pose_image': im_pose,  # for visualization
                
#                     'grid_image': im_g,     # for visualization
                
                    
#                     'im_h_l': im_h_l,
#                     'gt_cltoh_warp_mask' : gt_cltoh_warp_mask,
#                     # 'edge_gt_cltoh_warp': edge_gt_cltoh_warp,
#                     # 'cloth_edge':     ce,
#                     }
                
#             elif self.datamode == 'test':
#                 result = {
#                     'c_name':   c_name,     # for visualization
#                     'im_name':  im_name,    # for visualization or ground truth
                
#                     'cloth':    c,          # for input
#                     'cloth_mask':     cm,   # for input
                
#                     'image':    im,         # for visualization
                
#                     'agnostic': agnostic,   # for input
                
#                     'parse_cloth': im_c,    # for ground truth
#                     'shape': shape,         # for visualization
#                     'head': im_h,           # for visualization
#                     # 'pose_image': im_pose,  # for visualization
                
#                     'grid_image': im_g,     # for visualization
                
                    
#                     'im_h_l': im_h_l,
#                     # 'cloth_edge':     ce,
                    
#                     }
                
#         elif self.stage == 'TOM':
#             if self.datamode == 'train':
#                 result = {
#                 'c_name':   c_name,     # for visualization
#                 'im_name':  im_name,    # for visualization or ground truth
            
#                 'cloth':    c,          # for input
#                 # 'cloth_mask':     cm,   # for input
            
#                 'image':    im,         # for visualization
            
#                 'agnostic': agnostic,   # for input
            
#                 'parse_cloth': im_c,    # for ground truth
#                 'shape': shape,         # for visualization
#                 'head': im_h,           # for visualization
#                 'pose_image': im_pose,  # for visualization
            
#                 'grid_image': im_g,     # for visualization
                
#                 'gt_cltoh_warp_mask' : gt_cltoh_warp_mask,
                
#                 'im_h_l': im_h_l,
#                 "real_c" :real_c,
#                 # 'cloth_edge':     ce,
                
#                 }
#             if self.datamode == 'test':
#                 result = {
#                     'c_name':   c_name,     # for visualization
#                     'im_name':  im_name,    # for visualization or ground truth
                
#                     'cloth':    c,          # for input
#                     # 'cloth_mask':     cm,   # for input
                
#                     'image':    im,         # for visualization
                
#                     'agnostic': agnostic,   # for input
                
#                     'parse_cloth': im_c,    # for ground truth
#                     'shape': shape,         # for visualization
#                     'head': im_h,           # for visualization
#                     'pose_image': im_pose,  # for visualization
                
#                     'grid_image': im_g,     # for visualization
                
                    
#                     'im_h_l': im_h_l,
                    
#                     "real_c" :real_c,
#                     # 'cloth_edge':     ce,
                    
#                     }

#         return result

#     def __len__(self):
#         return len(self.im_names)

# class CPDataLoader(object):
#     def __init__(self, opt, dataset):
#         super(CPDataLoader, self).__init__()

#         if opt.shuffle :
#             train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
#         else:
#             train_sampler = None

#         self.data_loader = torch.utils.data.DataLoader(
#                 dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
#                 num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
#         self.dataset = dataset
#         self.data_iter = self.data_loader.__iter__()
       
#     def next_batch(self):
#         try:
#             batch = self.data_iter.__next__()
#         except StopIteration:
#             self.data_iter = self.data_loader.__iter__()
#             batch = self.data_iter.__next__()

#         return batch


# if __name__ == "__main__":
#     print("Check the dataset for geometric matching module!")
    
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataroot", default = "data")
#     parser.add_argument("--datamode", default = "train")
#     parser.add_argument("--stage", default = "GMM")
#     parser.add_argument("--data_list", default = "train_pairs.txt")
#     parser.add_argument("--fine_width", type=int, default = 384)
#     parser.add_argument("--fine_height", type=int, default = 512)
#     parser.add_argument("--radius", type=int, default = 3)
#     parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
#     parser.add_argument('-b', '--batch-size', type=int, default=4)
#     parser.add_argument('-j', '--workers', type=int, default=1)
    
#     opt = parser.parse_args()
#     dataset = CPDataset(opt)
#     data_loader = CPDataLoader(opt, dataset)

#     print('Size of the dataset: %05d, dataloader: %04d' \
#             % (len(dataset), len(data_loader.data_loader)))
#     first_item = dataset.__getitem__(0)
#     first_batch = data_loader.next_batch()

#     from IPython import embed; embed()





# # #coding=utf-8
# # import torch
# # import torch.utils.data as data
# # import torchvision.transforms as transforms

# # from PIL import Image
# # from PIL import ImageDraw

# # import os.path as osp
# # import numpy as np
# # import json

# # class CPDataset(data.Dataset):
# #     """Dataset for CP-VTON.
# #     """
# #     def __init__(self, opt):
# #         super(CPDataset, self).__init__()
# #         # base setting
# #         self.opt = opt
# #         self.root = opt.dataroot
# #         self.datamode = opt.datamode # train or test or self-defined
# #         self.stage = opt.stage # GMM or TOM
# #         self.data_list = opt.data_list
# #         self.fine_height = opt.fine_height
# #         self.fine_width = opt.fine_width
# #         self.radius = opt.radius
# #         self.data_path = osp.join(opt.dataroot, opt.datamode)
# #         self.transform = transforms.Compose([  \
# #                 transforms.ToTensor(),   \
# #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
# #         # load data list
# #         im_names = []
# #         c_names = []
# #         with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
# #             for line in f.readlines():
# #                 im_name, c_name = line.strip().split()
# #                 im_names.append(im_name)
# #                 c_names.append(c_name)

# #         self.im_names = im_names
# #         self.c_names = c_names

# #     def name(self):
# #         return "CPDataset"

# #     def __getitem__(self, index):
# #         c_name = self.c_names[index]
# #         im_name = self.im_names[index]

# #         # cloth image & cloth mask
# #         if self.stage == 'GMM':
# #             c = Image.open(osp.join(self.data_path, 'cloth', c_name))
# #             cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name))
# #         else:
# #             c = Image.open(osp.join(self.data_path, 'warp-cloth', c_name))
# #             cm = Image.open(osp.join(self.data_path, 'warp-mask', c_name))
     
# #         c = self.transform(c)  # [-1,1]
# #         cm_array = np.array(cm)
# #         cm_array = (cm_array >= 128).astype(np.float32)
# #         cm = torch.from_numpy(cm_array) # [0,1]
# #         cm.unsqueeze_(0)

# #         # person image 
# #         im = Image.open(osp.join(self.data_path, 'image', im_name))
# #         im = self.transform(im) # [-1,1]

# #         # load parsing image
# #         parse_name = im_name.replace('.jpg', '.png')
# #         im_parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
# #         parse_array = np.array(im_parse)
# #         parse_shape = (parse_array > 0).astype(np.float32)
# #         parse_head = (parse_array == 1).astype(np.float32) + \
# #                 (parse_array == 2).astype(np.float32) + \
# #                 (parse_array == 4).astype(np.float32) + \
# #                 (parse_array == 13).astype(np.float32)
# #         parse_cloth = (parse_array == 5).astype(np.float32) + \
# #                 (parse_array == 6).astype(np.float32) + \
# #                 (parse_array == 7).astype(np.float32)
       
# #         # shape downsample
# #         parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
# #         parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
# #         parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
# #         shape = self.transform(parse_shape) # [-1,1]
# #         phead = torch.from_numpy(parse_head) # [0,1]
# #         pcm = torch.from_numpy(parse_cloth) # [0,1]

# #         # upper cloth
# #         im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
# #         im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

# #         # load pose points
# #         pose_name = im_name.replace('.jpg', '_keypoints.json')
# #         with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
# #             pose_label = json.load(f)
# #             pose_data = pose_label['people'][0]['pose_keypoints']
# #             pose_data = np.array(pose_data)
# #             pose_data = pose_data.reshape((-1,3))

# #         point_num = pose_data.shape[0]
# #         pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
# #         r = self.radius
# #         im_pose = Image.new('L', (self.fine_width, self.fine_height))
# #         pose_draw = ImageDraw.Draw(im_pose)
# #         for i in range(point_num):
# #             one_map = Image.new('L', (self.fine_width, self.fine_height))
# #             draw = ImageDraw.Draw(one_map)
# #             pointx = pose_data[i,0]
# #             pointy = pose_data[i,1]
# #             if pointx > 1 and pointy > 1:
# #                 draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
# #                 pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
# #             one_map = self.transform(one_map)
# #             pose_map[i] = one_map[0]

# #         # just for visualization
# #         im_pose = self.transform(im_pose)
        
# #         # cloth-agnostic representation
# #         agnostic = torch.cat([shape, im_h, pose_map], 0) 

# #         if self.stage == 'GMM':
# #             im_g = Image.open('grid.png')
# #             im_g = self.transform(im_g)
# #         else:
# #             im_g = ''

# #         result = {
# #             'c_name':   c_name,     # for visualization
# #             'im_name':  im_name,    # for visualization or ground truth
# #             'cloth':    c,          # for input
# #             'cloth_mask':     cm,   # for input
# #             'image':    im,         # for visualization
# #             'agnostic': agnostic,   # for input
# #             'parse_cloth': im_c,    # for ground truth
# #             'shape': shape,         # for visualization
# #             'head': im_h,           # for visualization
# #             'pose_image': im_pose,  # for visualization
# #             'grid_image': im_g,     # for visualization
# #             }

# #         return result

# #     def __len__(self):
# #         return len(self.im_names)

# # class CPDataLoader(object):
# #     def __init__(self, opt, dataset):
# #         super(CPDataLoader, self).__init__()

# #         if opt.shuffle :
# #             train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
# #         else:
# #             train_sampler = None

# #         self.data_loader = torch.utils.data.DataLoader(
# #                 dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
# #                 num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
# #         self.dataset = dataset
# #         self.data_iter = self.data_loader.__iter__()
       
# #     def next_batch(self):
# #         try:
# #             batch = self.data_iter.__next__()
# #         except StopIteration:
# #             self.data_iter = self.data_loader.__iter__()
# #             batch = self.data_iter.__next__()

# #         return batch


# # if __name__ == "__main__":
# #     print("Check the dataset for geometric matching module!")
    
# #     import argparse
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--dataroot", default = "data")
# #     parser.add_argument("--datamode", default = "train")
# #     parser.add_argument("--stage", default = "GMM")
# #     parser.add_argument("--data_list", default = "train_pairs.txt")
# #     parser.add_argument("--fine_width", type=int, default = 192)
# #     parser.add_argument("--fine_height", type=int, default = 256)
# #     parser.add_argument("--radius", type=int, default = 3)
# #     parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
# #     parser.add_argument('-b', '--batch-size', type=int, default=4)
# #     parser.add_argument('-j', '--workers', type=int, default=1)
    
# #     opt = parser.parse_args()
# #     dataset = CPDataset(opt)
# #     data_loader = CPDataLoader(opt, dataset)

# #     print('Size of the dataset: %05d, dataloader: %04d' \
# #             % (len(dataset), len(data_loader.data_loader)))
# #     first_item = dataset.__getitem__(0)
# #     first_batch = data_loader.next_batch()

# #     from IPython import embed; embed()

