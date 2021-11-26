import json

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, utils
import torch.utils.data as data

from datasets import get_CIFAR10, get_SVHN, postprocess
from model import Glow

device = "cpu" if (not torch.cuda.is_available())  else "cuda:6"
output_folder = 'output/'
model_name = 'glow_checkpoint_10934.pt'

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

    image_shape, num_classes, _, test_cifar = get_CIFAR10(hparams['augment'], hparams['dataroot'], hparams['download'])
    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                hparams['learn_top'], hparams['y_condition'])
    model.load_state_dict(torch.load(output_folder + model_name))
    model.set_actnorm_init()
    model = model.to(device)
    model = model.eval()

    #recovery
    test_loader = data.DataLoader(test_cifar,batch_size=20,shuffle=False,num_workers=0,drop_last=False)

    with torch.no_grad():
        utils.save_image(
            model_single.reverse(z_sample).cpu().data,
            f"logs/glow/{str(1).zfill(6)}.png",
            normalize=True,
            nrow=10,
            range=(-0.5, 0.5),
        )

    #sample 
    images = sample(model, num_classes, hparams['batch_size'])
    grid = make_grid(images[:30], nrow=6).permute(1,2,0)
    plt.figure(figsize=(10,10))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(output_folder+'cifar_sample.png')

    #Manipulation in latent space

