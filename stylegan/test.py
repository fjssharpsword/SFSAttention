import torch
import os
import imageio

if __name__ == "__main__":
    #result = torch.load('/data/pycode/SFSAttention/stylegan/imgs/IDRiD_098.pt')
    #latent = result['imgs/IDRiD_098.jpg']['latent']
    #print(latent.shape)
    images = []
    for i in range(420001):
        if i % 10000 == 0:
            filename = f"/data/pycode/SFSAttention/stylegan/sample/{str(i).zfill(6)}.png"
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
    imageio.mimsave('/data/pycode/SFSAttention/stylegan/imgs/fundus_sample_stylegan.gif', images,fps=1)
