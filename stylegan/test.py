import torch

if __name__ == "__main__":
    result = torch.load('/data/pycode/SFSAttention/stylegan/imgs/IDRiD_098.pt')
    latent = result['imgs/IDRiD_098.jpg']['latent']
    print(latent.shape)
