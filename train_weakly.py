import torch
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F
import numpy as np
import os
import random
from dataset import PF_Pascal, Pascal_BD
from noise_loss import loss_functionw as loss_function
from model_noise import PMDNet
from early_stop import EarlyStopping
from noise_weights import NoiseModule
# import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="PMDNet")
parser.add_argument('--gpu', type=str, default='5', help='random seed')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.2, help='decaying factor')
parser.add_argument('--decay_schedule', type=str, default='60', help='learning rate decaying schedule')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--feature_h', type=int, default=20, help='height of feature volume')
parser.add_argument('--feature_w', type=int, default=20, help='width of feature volume')
parser.add_argument('--train_image_path', type=str, default='./data/PF_Pascal/',
                    help='directory of pre-processed(.npy) images')
parser.add_argument('--train_csv_path', type=str, default='./data/PF_Pascal/train_ps.csv',
                    help='directory of test csv file')
parser.add_argument('--valid_csv_path', type=str, default='./data/PF_Pascal/bbox_test_pairs_pf_pascal.csv',
                    help='directory of validation csv file')
parser.add_argument('--valid_image_path', type=str, default='./data/PF_Pascal/', help='directory of validation data')
parser.add_argument('--beta', type=float, default=50, help='inverse temperature of softmax @ kernel soft argmax')
parser.add_argument('--kernel_sigma', type=float, default=5,
                    help='standard deviation of Gaussian kerenl @ kernel soft argmax')
parser.add_argument('--lambda1', type=float, default=16, help='weight parameter of mask consistency loss')
parser.add_argument('--lambda2', type=float, default=16, help='weight parameter of flow consistency loss')
parser.add_argument('--lambda3', type=float, default=0.5, help='weight parameter of smoothness loss')
parser.add_argument('--eval_type', type=str, default='image_size', choices=('bounding_box', 'image_size'),
                    help='evaluation type for PCK threshold (bounding box | image size)')
parser.add_argument('--model_path', type=str, default='weakly')
parser.add_argument('--log_file', type=str, default='training_weights.txt')
parser.add_argument('--is_tps', type=bool, default=True)
parser.add_argument('--is_scale', type=bool, default=True)
parser.add_argument('--min_scale', type=float, default=0.75)
parser.add_argument('--max_scale', type=float, default=1.25)
parser.add_argument('--tps_scale', type=float, default=0.15)
args = parser.parse_args()
resume = False
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Set seed
if args.seed == None:
    args.seed = np.random.randint(10000)
print('Seed number: ', args.seed)
print('model path: ', args.model_path)
print('lr: ', args.lr)
print('total epochs: ', args.epochs)

global global_seed
global_seed = args.seed
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)
torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    seed = global_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

def update_full(model, train_loader, device = None):

    pred_model, noise_model = model
    for batch_idx, data in enumerate(train_loader):
        # x : input data, (None, 3, 128, 128)
        x1 = data['image1'].to(device)
        x2 = data['image2'].to(device)
        item_idxs = data['idx']
        # y_noise: Unsup labels, (None, NUM_MAPS, 128, 128)
        y_noise = data['flow_T2S'].to(device)
        # pred: (None, 1, 128, 128)
        output = pred_model(x1,x2,train=False)
        pred = output['flow_T2S'].repeat((1,n_maps,1,1,1))
        # emp_var: Emperical Variance for each pixel for each image, (None, 1, 128, 128)
        emp_var = torch.var(y_noise - pred, 1).view(args.batch_size, -1).detach().cpu()
        _, var_idx = noise_model.get_index_multiple(img_idxs=item_idxs)
        noise_model.emp_var[var_idx.reshape(-1)] = emp_var.view(-1).to(noise_model.emp_var.device)
    noise_model.update()

# Make a log file & directory for saving weights
def log(text, LOGGER_FILE):
    with open(LOGGER_FILE, 'a') as f:
        f.write(text)
        f.close()


LOGGER_FILE = args.log_file
model_path = args.model_path

if os.path.exists(LOGGER_FILE):
    os.remove(LOGGER_FILE)

if not os.path.exists(model_path):
    os.mkdir(model_path)

# Data Loader
train_dataset = Pascal_BD(args.train_csv_path, args.train_image_path, args.feature_h, args.feature_w, args, args.eval_type)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers = args.num_workers,
                                           worker_init_fn = _init_fn)
n_imgs = len(train_dataset)
n_maps = 4

valid_dataset = PF_Pascal(args.valid_csv_path, args.valid_image_path, args.feature_h, args.feature_w, args.eval_type)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=1,
                                           shuffle=False, num_workers=args.num_workers)

# Instantiate model
net = PMDNet(args.feature_h, args.feature_w, beta=args.beta, kernel_sigma=args.kernel_sigma)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

noise_model = NoiseModule(num_imgs=n_imgs, num_maps=n_maps, img_size=(args.feature_w, args.feature_h), device=device)
early_stopping = EarlyStopping(net, valid_loader, args.feature_w, args.feature_h, device=device)

if resume:
    print("Load pre-trained weights")
    best_weights = torch.load("./w1/best_checkpoint.pt")
    adap3_dict = best_weights['state_dict1']
    adap4_dict = best_weights['state_dict2']
    chn4_dict = best_weights['chn4_dict']
    net.adap_layer_feat3.load_state_dict(adap3_dict)
    net.adap_layer_feat4.load_state_dict(adap4_dict)
    net.chn4.load_state_dict(chn4_dict)

# Instantiate loss
criterion = loss_function(args).to(device)

# Instantiate optimizer
param = list(net.adap_layer_feat3.parameters()) +\
        list(net.adap_layer_feat4.parameters()) +\
        list(net.chn4.parameters())

optimizer = torch.optim.Adam(param, lr=args.lr)
decay_schedule = list(map(lambda x: int(x), args.decay_schedule.split('-')))
scheduler = lrs.MultiStepLR(optimizer, milestones=decay_schedule, gamma=args.gamma)

def correct_keypoints(source_points, warped_points, L_pck, alpha=0.1):
    # compute correct keypoints
    p_src = source_points[0, :]
    p_wrp = warped_points[0, :]

    N_pts = torch.sum(torch.ne(p_src[0, :], -1) * torch.ne(p_src[1, :], -1))
    point_distance = torch.pow(torch.sum(torch.pow(p_src[:, :N_pts] - p_wrp[:, :N_pts], 2), 0), 0.5)
    L_pck_mat = L_pck[0].expand_as(point_distance)
    correct_points = torch.le(point_distance, L_pck_mat * alpha)
    pck = torch.mean(correct_points.float())
    return pck


# Training
best_pck = 0
train_noise = 1
print('Training started')
for ep in range(args.epochs):
    print('Current epoch : %d' % ep)
    log('Current epoch : %d\n' % ep, LOGGER_FILE)
    log('Current learning rate : %e\n' % optimizer.state_dict()['param_groups'][0]['lr'], LOGGER_FILE)

    net.train()
    net.feature_extraction.eval()
    total_loss = 0

    for i, batch in enumerate(train_loader):
        src_image = batch['image1'].to(device)
        tgt_image = batch['image2'].to(device)
        flow_T2S = batch['flow_T2S'].to(device)
        mask_T2S = batch['mask_T2S'].to(device)
        bsize = src_image.shape[0]
        item_idxs = batch['idx']

        output = net(src_image, tgt_image)

        if not train_noise:
            pred = output['flow_T2S']
            weights = torch.ones_like(pred).to(device)
        else:
            noise_prior = noise_model.sample_noise(item_idxs).to(device)
            pred = output['flow_T2S']
            dists = flow_T2S - pred
            weights = torch.exp(-(dists**2)/(2*noise_prior**2).unsqueeze(1))

        noise_loss = 0.0
        if train_noise:
            emp_var = torch.var(flow_T2S - pred, 1).view(bsize, -1) + 1e-16
            prior_var, var_idx = noise_model.get_index_multiple(img_idxs=item_idxs)
            prior_var = torch.from_numpy(prior_var).float()
            # Important Order for loss var needs to get close to emp_var
            noise_loss = noise_model.loss_fast(prior_var, emp_var)

        optimizer.zero_grad()
        # mask_T2S = (torch.ge(mask_T2S, -10)).float()
        loss, L2, L3 = criterion(output, flow_T2S, mask_T2S, weights)
        loss += 0.01 * noise_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        log("Epoch %03d (%04d/%04d) = Loss : %5f (Now : %5f)\t" % (
        ep, i, len(train_dataset) // args.batch_size, total_loss / (i + 1), loss.cpu().data), LOGGER_FILE)
        print("Epoch %03d (%04d/%04d) = Loss : %5f (Now : %5f)\t" % (
        ep, i, len(train_dataset) // args.batch_size, total_loss / (i + 1), loss.cpu().data))
        log("L2 : %5f, L3 : %5f\n" % (L2.item(), L3.item()), LOGGER_FILE)
        print("L2 : %5f, L3 : %5f\n" % (L2.item(), L3.item()))

    early_stopping.validate()
    train_noise = 0
    if early_stopping.converged == 1 and train_noise == 0:
        # Update the noise variance using Eq 7. !Note: Importantly we do this for all images encountered, using the pred_variance
        update_full([net, noise_model], train_loader, device=device)
        print('Updating Noise Variance')
        train_noise = 1
        # Reset Early Stopping variables, training till next convergence
        early_stopping.reset()

    scheduler.step()
    log("Epoch %03d finished... Average loss : %5f\n" % (ep, total_loss / len(train_loader)), LOGGER_FILE)
    print("Epoch %03d finished... Average loss : %5f\n" % (ep, total_loss / len(train_loader)))

    with torch.no_grad():
        log('Computing PCK@Validation set...', LOGGER_FILE)
        print('Computing PCK@Validation set...')
        net.eval()
        total_correct_points = 0
        total_points = 0
        print(len(valid_loader))
        for i, batch in enumerate(valid_loader):
            src_image = batch['image1'].to(device)
            tgt_image = batch['image2'].to(device)
            output = net(src_image, tgt_image, train=False)

            small_grid = output['grid_T2S'][:, :, :, :]
            small_grid[:, :, :, 0] = small_grid[:, :, :, 0] * (args.feature_w // 2) / (args.feature_w // 2 - 0)
            small_grid[:, :, :, 1] = small_grid[:, :, :, 1] * (args.feature_h // 2) / (args.feature_h // 2 - 0)
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
                                                      torch.FloatTensor(est_image1_points).unsqueeze(0),
                                                      batch['L_pck'], alpha=0.1)
        PCK = total_correct_points / len(valid_dataset)
        log('PCK: %5f\n\n' % PCK, LOGGER_FILE)
        print('current PCK: %5f || best PCK: %.5f\n' % (PCK, best_pck))
        if PCK > best_pck:
            best_pck = PCK
            print('saving....{:}'.format(model_path + '/best_checkpoint.pt'))
            torch.save({'state_dict1': net.adap_layer_feat3.state_dict(),
                        'state_dict2': net.adap_layer_feat4.state_dict(),
                        'chn4_dict': net.chn4.state_dict()},
                         model_path + '/best_checkpoint.pt')
            noise_model.save(model_path + '/best_noise.npy')

print('Done')