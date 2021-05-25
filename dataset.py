import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import os
import transforms
import cv2
import json

class PF_Pascal(Dataset):
    def __init__(self, csv_path, image_path, feature_H, feature_W, eval_type='image_size'):
        self.feature_H = feature_H
        self.feature_W = feature_W

        self.image_H = (self.feature_H - 0) * 16
        self.image_W = (self.feature_W - 0) * 16

        self.data_info = pd.read_csv(csv_path)

        self.transform = transforms.Compose([transforms.Resize((self.image_H, self.image_W), interpolation=2),
                                             transforms.Pad(0),  # pad zeros around borders to avoid boundary artifacts
                                             transforms.ToTensor()])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_A_names = self.data_info.iloc[:, 0]
        self.image_B_names = self.data_info.iloc[:, 1]
        self.class_num = self.data_info.iloc[:, 2]
        self.point_A_coords = self.data_info.iloc[:, 3:5]
        self.point_B_coords = self.data_info.iloc[:, 5:7]
        self.L_pck = self.data_info.iloc[:, 7].values.astype('float')  # L_pck of source
        self.image_path = image_path
        self.eval_type = eval_type

    def get_image(self, image_name_list, idx):
        image_name = os.path.join(self.image_path, image_name_list[idx])
        image = Image.open(image_name)
        width, height = image.size
        return image, torch.FloatTensor([height, width])

    def get_points(self, point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=';')
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=';')
        point_coords = np.concatenate((X.reshape(1, len(X)), Y.reshape(1, len(Y))), axis=0)
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords
    def __getitem__(self, idx):
        # get pre-processed images
        image1, image1_size = self.get_image(self.image_A_names, idx)
        image2, image2_size = self.get_image(self.image_B_names, idx)
        class_num = int(self.class_num[idx]) - 1
        image1_var = self.transform(image1)
        image2_var = self.transform(image2)

        # get pre-processed point coords
        point_A_coords = self.get_points(self.point_A_coords, idx)
        point_B_coords = self.get_points(self.point_B_coords, idx)
        # compute PCK reference length L_pck (equal to max bounding box side in image_B)
        if self.eval_type == 'bounding_box':
            # L_pck = torch.FloatTensor([torch.max(point_B_coords.max(1)[0] - point_B_coords.min(1)[0])]) # for PF WILLOW
            L_pck = torch.FloatTensor(np.fromstring(self.L_pck[idx]).astype(
                np.float32))  # max(h,w), where h&w are height&width of bounding-box provided by Pascal dataset
        elif self.eval_type == 'image_size':
            N_pts = torch.sum(torch.ne(point_A_coords[0, :], -1))
            point_A_coords[0, 0:N_pts] = point_A_coords[0, 0:N_pts] * self.image_W / image1_size[1]  # rescale x coord.
            point_A_coords[1, 0:N_pts] = point_A_coords[1, 0:N_pts] * self.image_H / image1_size[0]  # rescale y coord.
            point_B_coords[0, 0:N_pts] = point_B_coords[0, 0:N_pts] * self.image_W / image2_size[1]  # rescale x coord.
            point_B_coords[1, 0:N_pts] = point_B_coords[1, 0:N_pts] * self.image_H / image2_size[0]  # rescale y coord.
            image1_size = torch.FloatTensor([self.image_H, self.image_W])
            image2_size = torch.FloatTensor([self.image_H, self.image_W])
            L_pck = torch.FloatTensor([self.image_H]) if self.image_H >= self.image_W else torch.FloatTensor(
                [self.image_W])
        else:
            raise ValueError('Invalid eval_type')

        return {'image1_rgb': transforms.ToTensor()(image1), 'image2_rgb': transforms.ToTensor()(image2),
                'image1': self.normalize(image1_var), 'image2': self.normalize(image2_var),
                'image1_points': point_A_coords, 'image2_points': point_B_coords, 'L_pck': L_pck,
                'image1_size': image1_size, 'image2_size': image2_size, 'class_num': class_num,
                'image1_name':self.image_A_names[idx].split('/')[-1][:-4],
                'image2_name':self.image_B_names[idx].split('/')[-1][:-4],
                }

    def __len__(self):
        return len(self.data_info.index)

class SPair(Dataset):
    def __init__(self, root_path, feature_H, feature_W, type='train' ,eval_type='image_size',):
        self.feature_H = feature_H
        self.feature_W = feature_W

        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.image_H = (self.feature_H - 0) * 16
        self.image_W = (self.feature_W - 0) * 16

        self.root_path = root_path
        self.json_path = root_path +'/PairAnnotation/'+ type
        self.json_names = os.listdir(self.json_path+'/')

        self.transform = transforms.Compose([transforms.Resize((self.image_H, self.image_W), interpolation=2),
                                             transforms.Pad(0),  # pad zeros around borders to avoid boundary artifacts
                                             transforms.ToTensor()])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.eval_type = eval_type

    def get_points(self, point_coords):
        X = point_coords[0,:]
        Y = point_coords[1,:]
        Xpad = -np.ones(20)
        Xpad[:len(X)] = X
        Ypad = -np.ones(20)
        Ypad[:len(X)] = Y
        point_coords = np.concatenate((Xpad.reshape(1, len(Xpad)), Ypad.reshape(1, len(Ypad))), axis=0)
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords

    def get_imagesfromjson(self, json_name):
        json_file = os.path.join(self.json_path, json_name)
        with open(json_file, 'r') as f:
            obj = json.load(f)
        src_image = Image.open(self.root_path+'/JPEGImages/'+obj['filename'].split(':')[-1]+'/'+obj['src_imname'])
        tgt_image = Image.open(self.root_path + '/JPEGImages/' + obj['filename'].split(':')[-1] + '/' + obj['trg_imname'])
        class_num = self.class_names.index(obj['category'])
        width, height = src_image.size
        src_size = torch.FloatTensor([height, width])
        width, height = tgt_image.size
        tgt_size = torch.FloatTensor([height, width])

        point_A_coords = self.get_points(np.array(obj['src_kps'], np.float32).T)
        point_A_coords = torch.Tensor(point_A_coords)
        point_B_coords = self.get_points(np.array(obj['trg_kps'], np.float32).T)
        point_B_coords = torch.Tensor(point_B_coords)

        if self.eval_type == 'bounding_box':
            trg_bbox = obj['trg_bndbox']
            src_bbox = obj['src_bndbox']
            L_pck = torch.FloatTensor([max(src_bbox[2] - src_bbox[0], src_bbox[3] - src_bbox[1])])
        elif self.eval_type == 'image_size':
            N_pts = torch.sum(torch.ne(point_A_coords[0, :], -1))
            point_A_coords[0, 0:N_pts] = point_A_coords[0, 0:N_pts] * self.image_W / src_size[1]  # rescale x coord.
            point_A_coords[1, 0:N_pts] = point_A_coords[1, 0:N_pts] * self.image_H / src_size[0]  # rescale y coord.
            point_B_coords[0, 0:N_pts] = point_B_coords[0, 0:N_pts] * self.image_W / tgt_size[1]  # rescale x coord.
            point_B_coords[1, 0:N_pts] = point_B_coords[1, 0:N_pts] * self.image_H / tgt_size[0]  # rescale y coord.
            src_size = torch.FloatTensor([self.image_H, self.image_W])
            tgt_size = torch.FloatTensor([self.image_H, self.image_W])
            L_pck = torch.FloatTensor([self.image_H]) if self.image_H >= self.image_W else torch.FloatTensor(
                [self.image_W])
        else:
            raise ValueError('Invalid eval_type')

        return src_image, tgt_image, src_size, tgt_size, \
               point_A_coords, point_B_coords, L_pck, \
               obj['src_imname'][:-4], obj['trg_imname'][:-4], class_num

    def __getitem__(self, idx):
        # get pre-processed images
        image1, image2, image1_size, image2_size, \
             image1_points, image2_points, L_pck, \
                            image1_name, image2_name, class_num = self.get_imagesfromjson(self.json_names[idx])

        image1_var = self.transform(image1)
        image2_var = self.transform(image2)

        return {'image1': self.normalize(image1_var), 'image2': self.normalize(image2_var),
                'image1_points': image1_points, 'image2_points': image2_points, 'L_pck': L_pck,
                'image1_size': image1_size, 'image2_size': image2_size, 'class_num': class_num,
                'image1_name': image1_name, 'image2_name': image2_name, 'idx': idx,
                'image1_rgb': transforms.ToTensor()(image1), 'image2_rgb': transforms.ToTensor()(image2)}
    def __len__(self):
        return len(self.json_names)

class PF_WILLOW(Dataset):
    def __init__(self, csv_path, image_path, feature_H, feature_W, eval_type='bounding_box'):
        self.feature_H = feature_H
        self.feature_W = feature_W

        self.image_H = (self.feature_H - 0) * 16
        self.image_W = (self.feature_W - 0) * 16

        self.data_info = pd.read_csv(csv_path)

        self.transform = transforms.Compose([transforms.Resize((self.image_H, self.image_W), interpolation=2),
                                             transforms.Pad(0),  # pad zeros around borders to avoid boundary artifacts
                                             transforms.ToTensor()])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_A_names = self.data_info.iloc[:, 0]
        self.image_B_names = self.data_info.iloc[:, 1]
        self.point_A_coords = self.data_info.iloc[:, 2:22].values.astype('float')
        self.point_B_coords = self.data_info.iloc[:, 22:].values.astype('float')
        self.L_pck = self.data_info.iloc[:, 7].values.astype('float')  # L_pck of source
        self.image_path = image_path
        self.eval_type = eval_type

    def get_image(self, image_name_list, idx):
        image_name = os.path.join(self.image_path, image_name_list[idx])
        image = Image.open(image_name)
        width, height = image.size
        return image, torch.FloatTensor([height, width])

    def get_points(self, point_coords_list, idx):
        point_coords = point_coords_list[idx, :].reshape(2, 10)
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords

    def __getitem__(self, idx):
        # get pre-processed images
        image1, image1_size = self.get_image(self.image_A_names, idx)
        image2, image2_size = self.get_image(self.image_B_names, idx)
        image1_var = self.transform(image1)
        image2_var = self.transform(image2)

        # get pre-processed point coords
        point_A_coords = self.get_points(self.point_A_coords, idx)
        point_B_coords = self.get_points(self.point_B_coords, idx)
        # compute PCK reference length L_pck (equal to max bounding box side in image_B)
        if self.eval_type == 'bounding_box':
            L_pck = torch.FloatTensor([torch.max(point_A_coords.max(1)[0] - point_A_coords.min(1)[0])])  # for PF WILLOW
        elif self.eval_type == 'image_size':
            N_pts = torch.sum(torch.ne(point_A_coords[0, :], -1))
            point_A_coords[0, 0:N_pts] = point_A_coords[0, 0:N_pts] * self.image_W / image1_size[1]  # rescale x coord.
            point_A_coords[1, 0:N_pts] = point_A_coords[1, 0:N_pts] * self.image_H / image1_size[0]  # rescale y coord.
            point_B_coords[0, 0:N_pts] = point_B_coords[0, 0:N_pts] * self.image_W / image2_size[1]  # rescale x coord.
            point_B_coords[1, 0:N_pts] = point_B_coords[1, 0:N_pts] * self.image_H / image2_size[0]  # rescale y coord.
            image1_size = torch.FloatTensor([self.image_H, self.image_W])
            image2_size = torch.FloatTensor([self.image_H, self.image_W])
            L_pck = torch.FloatTensor([self.image_H]) if self.image_H >= self.image_W else torch.FloatTensor(
                [self.image_W])
        else:
            raise ValueError('Invalid eval_type')

        return {'image1_rgb': transforms.ToTensor()(image1), 'image2_rgb': transforms.ToTensor()(image2),
                'image1': self.normalize(image1_var), 'image2': self.normalize(image2_var),
                'image1_points': point_A_coords, 'image2_points': point_B_coords, 'L_pck': L_pck,
                'image1_size': image1_size, 'image2_size': image2_size}

    def __len__(self):
        return len(self.data_info.index)

class Pascal_BD(Dataset):
    def __init__(self, csv_path, image_path, feature_H, feature_W, args, eval_type='image_size'):
        self.feature_H = feature_H
        self.feature_W = feature_W

        self.image_H = (self.feature_H - 0) * 16
        self.image_W = (self.feature_W - 0) * 16

        self.args = args

        self.image_X, self.image_Y = np.meshgrid(np.linspace(-1, 1, self.image_W),
                                               np.linspace(-1, 1, self.image_H))

        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, self.feature_W),
                                               np.linspace(-1, 1, self.feature_H))

        self.data_info = pd.read_csv(csv_path)

        self.transform = transforms.Compose([transforms.Resize((self.image_H, self.image_W), interpolation=2),
                                             transforms.Pad(0),  # pad zeros around borders to avoid boundary artifacts
                                             transforms.ToTensor()])

        self.mask_transform = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize((feature_H, feature_W)),
                                                   transforms.ToTensor()])

        self.image_transform = transforms.Compose([transforms.ToPILImage(),
            transforms.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor()])

        self.earse = transforms.Compose([transforms.RandomErasing(probability=0.3)])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_A_names = self.data_info.source_image
        self.image_B_names = self.data_info.target_image
        self.T2S_mask = self.data_info.T2S_mask
        self.S2T_mask = self.data_info.S2T_mask
        self.flow_T2S = self.data_info.flow_T2S
        self.flow_S2T = self.data_info.flow_S2T
        self.classes = self.data_info.classes
        self.flip = self.data_info.flip
        self.image_path = image_path
        self.eval_type = eval_type

    def affine_transform(self, x, theta):
        x = x.unsqueeze(0)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x[0], grid

    def get_image(self, image_name_list, idx, flip_list):
        image_name = os.path.join(self.image_path, image_name_list[idx])
        image = Image.open(image_name)
        if flip_list[idx]:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        width, height = image.size
        imname = image_name.split('/')[-1][:-4]
        return image, torch.FloatTensor([height, width]), imname

    def get_flow(self, flow_name_list, idx, flip_list):
        flow_names = flow_name_list[idx].split('|')[:-1]
        flow_array = []
        for i in range(len(flow_names)):
            flow_path = os.path.join(self.image_path, flow_names[i])
            flow_arr = np.load(flow_path)[0].astype(np.float32).transpose(1,2,0)
            if flip_list[idx]:
                flow_arr = flow_arr[:, ::-1, :]
                flow_arr[:, :, 0] = -flow_arr[:, :, 0]

            flow_arr = cv2.resize(flow_arr, (self.feature_W, self.feature_H), interpolation=cv2.INTER_LINEAR)
            flow_array.append(np.expand_dims(flow_arr, axis=0))
        flow_array = np.concatenate(flow_array, axis=0)
        return flow_array

    def get_masks(self, mask_name_list, idx, flip_list):
        mask_names = mask_name_list[idx].split('|')[:-1]
        mask_array = []
        for i in range(len(mask_names)):
            mask_path = os.path.join(self.image_path, mask_names[i])
            mask = Image.open(mask_path).convert('L')
            if flip_list[idx]:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.array(mask, np.float)
            mask = cv2.resize(mask,(self.feature_W, self.feature_H), interpolation=cv2.INTER_NEAREST)
            mask_array.append(np.expand_dims(mask, axis=0))
        mask_array = np.concatenate(mask_array, axis=0)
        mask_array = np.expand_dims(mask_array, axis=1)
        return mask_array

    def add_fwtps(self, flows, tps_flow):
        mask_arr = []
        for i in range(flows.shape[0]):
            flow_arr = flows[i,:,:,:]
            tps_flow, mask = self.warp_tps(flow_arr, tps_flow)
            flows[i,:,:,:] += tps_flow
            mask_arr.append(np.expand_dims(mask, axis=0))
        mask_arr = np.concatenate(mask_arr, axis=0)
        mask_arr = np.expand_dims(mask_arr, axis=1)
        return flows, mask_arr

    def add_bwtps(self, flows, tps_flow):
        mask_arr = []
        for i in range(flows.shape[0]):
            flow_arr = flows[i,:,:,:]
            flow_arr, mask = self.warp_tps(tps_flow, flow_arr)
            flows[i,:,:,:] = flow_arr + tps_flow
            mask_arr.append(np.expand_dims(mask, axis=0))
        mask_arr = np.concatenate(mask_arr, axis=0)
        mask_arr = np.expand_dims(mask_arr, axis=1)
        return flows, mask_arr

    def get_tps_flow(self, tps_grid):
        grid = np.concatenate((np.expand_dims(self.image_X, axis=-1),
                               np.expand_dims(self.image_Y, axis=-1)), axis=-1)
        grid = torch.from_numpy(grid).float()
        tps_flow = tps_grid[0] - grid
        tps_flow = tps_flow.unsqueeze(0).permute(0,3,1,2)
        tps_flow = F.interpolate(tps_flow,(self.feature_H, self.feature_W), mode='bilinear', align_corners=True)
        tps_flow = tps_flow.permute(0,2,3,1)
        return tps_flow.numpy()[0]

    def warp_tps(self, flow_arr, tps_flow):
        grid = np.concatenate((np.expand_dims(self.grid_X, axis=-1),
                               np.expand_dims(self.grid_Y, axis=-1)), axis=-1)
        grid = torch.from_numpy(grid).float()
        flow_arr = torch.from_numpy(flow_arr).float()
        tps_flow = torch.from_numpy(tps_flow).permute(2,0,1).unsqueeze(0)
        grid_arr = flow_arr + grid

        np_grid_arr = grid_arr.numpy()
        mask = (np_grid_arr[:,:,0]>=-1)*(np_grid_arr[:,:,0]<=1)*(np_grid_arr[:,:,1]>=-1)*(np_grid_arr[:,:,1]<=1)
        mask = mask.astype(np.float)

        grid_arr = grid_arr.unsqueeze(0)
        tps_flow = F.grid_sample(tps_flow, grid_arr, mode='bilinear')

        return tps_flow.permute(0, 2, 3, 1)[0].numpy(), mask

    def masks_transform(self, masks, affine):
        mask_array = []
        for i in range(masks.shape[0]):
            mask = torch.from_numpy(masks[0]).float()
            mask, _ = self.affine_transform(mask, affine)
            mask = (mask > 0.5).float() * 255
            mask_array.append(mask.numpy())
        mask_array = np.concatenate(mask_array, axis=0)
        mask_array = np.expand_dims(mask_array, axis=1)
        return mask_array

    def cycle_weights(self,flow_S2T,flow_T2S):
        grid = np.concatenate((np.expand_dims(self.grid_X, axis=-1),
                               np.expand_dims(self.grid_Y, axis=-1)), axis=-1)
        grid = torch.from_numpy(grid).float()
        flow_T2S = torch.from_numpy(flow_T2S)
        grid_T2S = flow_T2S + grid.unsqueeze(0)
        flow_S2T = torch.from_numpy(flow_S2T)
        warped_flow_T2S = -F.grid_sample(flow_S2T.permute(0,3,1,2), grid_T2S,mode='bilinear').permute(0,2,3,1)
        threshold = 0.08
        b = 50
        dists = torch.sum(torch.pow(warped_flow_T2S - flow_T2S, 2), dim=-1)
        dists = torch.sqrt(dists)
        dists = dists.unsqueeze(1).cpu().numpy()
        cycle_arr = 1 - 1.0 / (1.0 + np.exp(-b * (dists - threshold)))
        cycle_arr = (cycle_arr - np.min(cycle_arr)) / (np.max(cycle_arr) - np.min(cycle_arr))
        return cycle_arr

    def __getitem__(self, idx):
        image1, image1_size, image1_name = self.get_image(self.image_A_names, idx, self.flip)
        image2, image2_size, image2_name = self.get_image(self.image_B_names, idx, self.flip)
        # mask_S2T = self.get_masks(self.S2T_mask, idx, self.flip)
        # mask_T2S = self.get_masks(self.T2S_mask, idx, self.flip)
        image1_var = self.transform(image1)
        image2_var = self.transform(image2)
        image1 = image1_var.permute(1, 2, 0).numpy() * 255
        image2 = image2_var.permute(1, 2, 0).numpy() * 255
        image1_var = self.image_transform(image1_var)
        image2_var = self.image_transform(image2_var)

        image2_var = self.earse(image2_var)
        flow_T2S_var = self.get_flow(self.flow_T2S, idx, self.flip)
        flow_S2T_var = self.get_flow(self.flow_S2T, idx, self.flip)

        mask_T2S = self.cycle_weights(flow_S2T_var, flow_T2S_var)

        mask1 = np.ones((4, 1, self.feature_W, self.feature_H), np.float)
        # mask2 = np.ones((4, 1, self.feature_W, self.feature_H), np.float)
        if self.args.is_tps:
            theta1 = np.zeros(9)
            theta1[0:6] = np.random.randn(6) * self.args.tps_scale
            theta1 = theta1 + np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
            affine1 = np.reshape(theta1, (3, 3))
            affine_inverse1 = np.linalg.inv(affine1)
            affine1 = np.reshape(affine1, -1)[0:6]
            affine_inverse1 = np.reshape(affine_inverse1, -1)[0:6]
            affine1 = torch.from_numpy(affine1).type(torch.FloatTensor)
            affine_inverse1 = torch.from_numpy(affine_inverse1).type(torch.FloatTensor)
            image1_var, S2T_grid = self.affine_transform(image1_var, affine1)
            image1 = image1_var.permute(1, 2, 0).numpy() * 255
            _, T2S_grid = self.affine_transform(image1_var, affine_inverse1)
            T2S_flow = self.get_tps_flow(T2S_grid)
            flow_T2S_var, mask1 = self.add_fwtps(flow_T2S_var, T2S_flow)

        if self.args.is_scale:
            p = np.random.uniform()
            if p>0.2:
                xscale = np.random.uniform(self.args.min_scale,self.args.max_scale)
                yscale = np.random.uniform(self.args.min_scale,self.args.max_scale)
                theta1 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], np.float32)
                theta1[0:3] = theta1[0:3] * xscale
                theta1[3:6] = theta1[3:6] * yscale
                affine1 = np.reshape(theta1, (3, 3))
                affine_inverse1 = np.linalg.inv(affine1)
                affine1 = np.reshape(affine1, -1)[0:6]
                affine_inverse1 = np.reshape(affine_inverse1, -1)[0:6]
                affine1 = torch.from_numpy(affine1).type(torch.FloatTensor)
                affine_inverse1 = torch.from_numpy(affine_inverse1).type(torch.FloatTensor)
                image1_var, _ = self.affine_transform(image1_var, affine1)
                image1 = image1_var.permute(1, 2, 0).numpy() * 255
                _, T2S_grid = self.affine_transform(image1_var, affine_inverse1)
                T2S_flow = self.get_tps_flow(T2S_grid)
                flow_T2S_var, mask1 = self.add_fwtps(flow_T2S_var, T2S_flow)

        mask_T2S = mask1 * mask_T2S
        flow_T2S_var = torch.from_numpy(flow_T2S_var).permute(0, 3, 1, 2).float()

        mask_T2S = torch.from_numpy(mask_T2S).float()

        return {'image1_rgb': transforms.ToTensor()(image1), 'image2_rgb': transforms.ToTensor()(image2),
                'image1': self.normalize(image1_var), 'image2': self.normalize(image2_var),
                'flow_T2S': flow_T2S_var, 'cls': self.classes[idx],
                'mask_T2S':mask_T2S,
                'image1_size': image1_size, 'image2_size': image2_size,
                'image1_name': image1_name, 'image2_name': image2_name, 'idx': idx}

    def __len__(self):
        return len(self.data_info.index)