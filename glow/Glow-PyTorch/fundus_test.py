import json
import sys
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, utils
import torch.utils.data as data
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import numpy as np

from datasets import get_CIFAR10, get_Fundus, postprocess
from model import Glow

device = "cpu" if (not torch.cuda.is_available())  else "cuda:2"
output_folder = "/data/pycode/SFSAttention/glow/Glow-PyTorch/output/"

def sample(model, num_classes, batch_size):
    with torch.no_grad():
        if hparams['y_condition']:
            y = torch.eye(num_classes)
            y = y.repeat(batch_size // num_classes + 1)
            y = y[:32, :].to(device) # number hardcoded in model for now
        else:
            y = None

        images = postprocess(model(y_onehot=y, temperature=1, reverse=True))

    return images.cpu()

if __name__ == "__main__":

    with open(output_folder + 'hparams.json') as json_file:  
        hparams = json.load(json_file)

    image_shape, num_classes, train_fundus, test_fundus = get_Fundus()
    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                hparams['learn_top'], hparams['y_condition'])
    model.load_state_dict(torch.load(output_folder + 'glow_checkpoint_16274.pt', map_location={'cuda:7': 'cuda:1'})['model']) #model and optimizer
    model.set_actnorm_init()
    model = model.to(device)
    model = model.eval()
    
    #recovery
    test_loader = data.DataLoader(test_fundus, batch_size=4, shuffle=False,num_workers=0,drop_last=False)
    with torch.no_grad():
        for batch_idx, (img, lbl) in enumerate(test_loader):
            z_out, _, _ = model(x=img.to(device))
            img_p = model(z=z_out, temperature=1, reverse=True)
            img = torch.cat((img, img_p.cpu().data), 0)
            utils.save_image(
                img,
                f"/data/pycode/SFSAttention/glow/Glow-PyTorch/logs/fundus_recover.png",
                normalize=True,
                nrow=4,
                range=(-0.5, 0.5),
            )
            break;

    #sample 
    images = sample(model, num_classes, 10)
    grid = make_grid(images[:20], nrow=5).permute(1,2,0)
    plt.figure(figsize=(10,10))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig("/data/pycode/SFSAttention/glow/Glow-PyTorch/logs/fundus_sample.png")
    """
    #Manipulation in latent space
    x_mdr = torch.FloatTensor() #1
    x_sdr = torch.FloatTensor() #3
    test_loader = data.DataLoader(test_fundus,batch_size=1,shuffle=False,num_workers=0,drop_last=False)
    b_size = 2
    for batch_idx, (img, lbl) in enumerate(test_loader):
        if x_mdr.shape[0] == b_size and x_sdr.shape[0] == b_size: break
        if torch.argmax(lbl[0])==1.0 and x_mdr.shape[0] < b_size:
            x_mdr = torch.cat((x_mdr, img), 0)
        elif torch.argmax(lbl[0])==3.0 and x_sdr.shape[0] < b_size:
            x_sdr = torch.cat((x_sdr, img), 0)
        else: continue
    z_mdr, _, _ = model(x=x_mdr.to(device))
    z_sdr, _, _ = model(x=x_sdr.to(device))
    z_mdr = torch.mean(z_mdr, dim=0) #average and reduction
    z_sdr = torch.mean(z_sdr, dim=0)
    z_manipulate = z_mdr - z_sdr# Get manipulation vector by taking difference
    z_manipulate = z_manipulate.expand([x_mdr.shape[0], -1, -1, -1])
    z_out, _, _ = model(x=x_mdr.to(device)) 
    for alpha in [0.1, 0.3, 0.5, 0.7, 1.0]:
        z_manipulated = z_out + alpha * z_manipulate
        x_manipulated = model(z=z_manipulated, temperature=1, reverse=True)  
        utils.save_image(
                torch.cat([x_mdr, x_sdr, x_manipulated.cpu().data], 0),
                f"/data/pycode/SFSAttention/glow/Glow-PyTorch/logs/fundus_{str(alpha)}.png",
                normalize=True,
                nrow=8,
                range=(-0.5, 0.5),
            )
    """
    #for retrieval evaluation
    print('********************Build feature for trainset!********************')
    train_loader = data.DataLoader(train_fundus,batch_size=hparams['eval_batch_size'],shuffle=True,num_workers=0,drop_last=True)
    tr_label = torch.FloatTensor()
    tr_feat = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(train_loader):
            tr_label = torch.cat((tr_label, label), 0)
            var_feat, _, _ = model(x=image.to(device))
            tr_feat = torch.cat((tr_feat, var_feat.cpu().data.view(var_feat.shape[0],-1)), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    print('********************Extract feature for testset!********************')
    test_loader = data.DataLoader(test_fundus,batch_size=hparams['eval_batch_size'],shuffle=False,num_workers=0,drop_last=False)
    te_label = torch.FloatTensor()
    te_feat = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            te_label = torch.cat((te_label, label), 0)
            var_feat, _, _ = model(x=image.to(device))
            te_feat = torch.cat((te_feat, var_feat.cpu().data.view(var_feat.shape[0],-1)), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
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

        

