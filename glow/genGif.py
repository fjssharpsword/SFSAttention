import imageio
import os

if __name__ == "__main__":

    images = []
    for i in range(3700):
        if i % 200 == 0:
            filename = f"/data/pycode/SFSAttention/glow/GlowPyTorch/logs/{str(i + 1).zfill(6)}.png"
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
    imageio.mimsave('/data/pycode/SFSAttention/glow/GlowPyTorch/logs/cxr_sample.gif', images,fps=1)