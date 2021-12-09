import argparse
import math
import os
import sys
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import lpips
from model import Generator
from fundus_data import get_train_dataset_fundus, get_test_dataset_fundus

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


if __name__ == "__main__":
    device = "cuda:7"
    pro_root = "/data/pycode/SFSAttention/stylegan/"
    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    #parser.add_argument(
    #    "--ckpt", type=str, required=True, help="path to the model checkpoint"
    #)
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--mse", type=float, default=0, help="weight of the mse loss")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    #parser.add_argument(
    #    "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    #)

    args = parser.parse_args()

    n_mean_latent = 10000

    resize = min(args.size, 256)
    """
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    #load fundus images
    imgs, files = [], []
    PATH_TO_IMAGES_DIR_TRAIN = '/data/fjsdata/fundus/IDRID/BDiseaseGrading/OriginalImages/TrainingSet/'
    for root, dirs, files in os.walk(PATH_TO_IMAGES_DIR_TRAIN):
            for file in files:
                if 'jpg' in file:
                    file_path = os.path.join(root + file)
                    img = transform(Image.open(file_path).convert("RGB"))
                    imgs.append(img)
                    files.append(os.path.splitext(file)[0])
   
    PATH_TO_IMAGES_DIR_TEST = '/data/fjsdata/fundus/IDRID/BDiseaseGrading/OriginalImages/TestingSet/'
    for root, dirs, files in os.walk(PATH_TO_IMAGES_DIR_TEST):
            for file in files:
                file_path = os.path.join(root + file)
                img = transform(Image.open(imgfile).convert("RGB"))
                imgs.append(img)
                files.append(os.path.basename(file))
    
    imgs = torch.stack(imgs, 0).to(device)
    """
    #load model
    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(pro_root+'checkpoint/400000.pt', map_location={'cuda:0': 'cuda:7'})["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device) #generate z of 10000 with 512 lengths
        latent_out = g_ema.style(noise_sample) #FC layer, turn to style vectors

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss( model="net-lin", net="vgg", use_gpu=device.startswith("cuda"), gpu_ids=[7])

    print('********************Build feature for trainset!********************')
    data_loader = torch.utils.data.DataLoader(dataset=get_train_dataset_fundus(), batch_size=8,shuffle=False, num_workers=0)
    tr_label = torch.FloatTensor()
    tr_feat = torch.FloatTensor()
    for batch_idx, (imgs, lbls) in enumerate(data_loader):

        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        pbar = tqdm(range(args.step))
        latent_path = []

        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, imgs.to(device)).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, imgs.to(device))

            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)
            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )

        #save latent vectors and lables
        tr_feat = torch.cat((tr_feat, latent_in.cpu().data), 0)
        tr_label = torch.cat((tr_label, lbls), 0)

    print('********************Extract feature for testset!********************')
    data_loader = torch.utils.data.DataLoader(dataset=get_test_dataset_fundus(), batch_size=8,shuffle=False, num_workers=0)
    te_label = torch.FloatTensor()
    te_feat = torch.FloatTensor()
    for batch_idx, (imgs, lbls) in enumerate(data_loader):

        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        pbar = tqdm(range(args.step))
        latent_path = []

        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, imgs.to(device)).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, imgs.to(device))

            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)
            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )

        #save latent vectors and lables
        te_feat = torch.cat((te_feat, latent_in.cpu().data), 0)
        te_label = torch.cat((te_label, lbls), 0)
    
    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_feat.numpy(), tr_feat.numpy())
    te_label = te_label.numpy()
    tr_label = tr_label.numpy()

    for topk in [5, 10, 20]:
        mHRs = {0: [], 1: [], 2: [], 3: [], 4: []}  # Hit Ratio
        mHRs_avg = []
        mAPs = {0: [], 1: [], 2: [], 3: [], 4: []}  # mean average precision
        mAPs_avg = []
        # NDCG: lack of ground truth ranking labels
        for i in range(sim_mat.shape[0]):
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i, :].tolist()), key=lambda x: x[1]))
            num_pos = 0
            rank_pos = 0
            mAP = []
            te_idx = np.where(te_label[i, :] == 1)[0][0]
            for j in idxs:
                rank_pos = rank_pos + 1
                tr_idx = np.where(tr_label[j, :] == 1)[0][0]
                if tr_idx == te_idx:  # hit
                    num_pos = num_pos + 1
                    mAP.append(num_pos / rank_pos)
                else:
                    mAP.append(0)
            if len(mAP) > 0:
                mAPs[te_idx].append(np.mean(mAP))
                mAPs_avg.append(np.mean(mAP))
            else:
                mAPs[te_idx].append(0)
                mAPs_avg.append(0)
            mHRs[te_idx].append(num_pos / rank_pos)
            mHRs_avg.append(num_pos / rank_pos)
            sys.stdout.write('\r test set process: = {}'.format(i + 1))
            sys.stdout.flush()

        CLASS_NAMES = ['Normal', "Mild NPDR", 'Moderate NPDR', 'Severe NPDR', 'PDR']
        # Hit ratio
        for i in range(len(CLASS_NAMES)):
            print('Fundus mHR of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mHRs[i])))
        print("Fundus Average mHR@{}={:.4f}".format(topk, np.mean(mHRs_avg)))
        # average precision
        for i in range(len(CLASS_NAMES)):
            print('Fundus mAP of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mAPs[i])))
        print("Fundus Average mAP@{}={:.4f}".format(topk, np.mean(mAPs_avg)))

    #nohup python fundus_pro.py > logs/train.log 2>&1 &
    # python fundus_pro.py --step=2