import torch
import torch.nn.functional as F
import numpy as np
def correct_keypoints(source_points, warped_points, L_pck, alpha=0.1):
    p_src = source_points[0,:]
    p_wrp = warped_points[0,:]

    N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
    point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
    L_pck_mat = L_pck[0].expand_as(point_distance)
    correct_points = torch.le(point_distance, L_pck_mat * alpha)
    pck = torch.mean(correct_points.float())
    return pck

class EarlyStopping:
    def __init__(self, model, val_loader, feature_w, feature_h, device):
        self.model = model
        self.val_loader = val_loader
        self.best_f1 = 0
        self.paitence = 0
        self.wait = 0
        self.f1 = 0
        self.device = device
        self.converged = False
        self.feature_w = feature_w
        self.feature_h = feature_h

    def reset(self):
        self.converged = False
        self.wait = 0

    def check_cgt(self):
        if self.best_f1 is None:
            self.best_f1 = self.f1
        elif self.best_f1 < self.f1:
            if self.wait >= self.paitence:
                self.converged = True
                self.wait = 0
            else:
                self.wait += 1
        else:
            self.wait = 0
            self.best_f1 = self.f1
        print('current f1=%.4f | best f1=%.4f' % (self.f1, self.best_f1))
        print('wait %d/%d' % (self.wait, self.paitence))

    def validate(self):
        self.model.eval()
        total_correct_points = 0
        total_points = 0
        for i, batch in enumerate(self.val_loader):
            src_image = batch['image1'].to(self.device)
            tgt_image = batch['image2'].to(self.device)
            output = self.model(src_image, tgt_image, train=False)

            small_grid = output['grid_T2S'][:, :, :, :]
            small_grid[:, :, :, 0] = small_grid[:, :, :, 0] * (self.feature_w // 2) / (self.feature_w // 2 - 0)
            small_grid[:, :, :, 1] = small_grid[:, :, :, 1] * (self.feature_h // 2) / (self.feature_h // 2 - 0)
            src_image_H = int(batch['image1_size'][0][0])
            src_image_W = int(batch['image1_size'][0][1])
            tgt_image_H = int(batch['image2_size'][0][0])
            tgt_image_W = int(batch['image2_size'][0][1])
            small_grid = small_grid.permute(0, 3, 1, 2)
            grid = F.interpolate(small_grid, size=(tgt_image_H, tgt_image_W), mode='bilinear', align_corners=True)
            grid = grid.permute(0, 2, 3, 1)
            grid_np = grid.cpu().data.numpy()

            image1_points = batch['image1_points'][0]
            image2_points = batch['image2_points'][0]

            est_image1_points = np.zeros((2, image1_points.size(1)))
            for j in range(image2_points.size(1)):
                point_x = int(np.round(image2_points[0, j]))
                point_y = int(np.round(image2_points[1, j]))

                if point_x == -1 and point_y == -1:
                    continue

                if point_x == tgt_image_W:
                    point_x = point_x - 1

                if point_y == tgt_image_H:
                    point_y = point_y - 1

                est_y = (grid_np[0, point_y, point_x, 1] + 1) * (src_image_H - 1) / 2
                est_x = (grid_np[0, point_y, point_x, 0] + 1) * (src_image_W - 1) / 2
                est_image1_points[:, j] = [est_x, est_y]

            total_correct_points += correct_keypoints(batch['image1_points'],
                                                      torch.FloatTensor(est_image1_points).unsqueeze(0), batch['L_pck'],
                                                      alpha=0.1)
        self.f1 = total_correct_points / len(self.val_loader)
        print('validation f1=%.4f'%(self.f1))
        self.check_cgt()
