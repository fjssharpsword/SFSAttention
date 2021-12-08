
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from parallel_wavegan.utils import load_model
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from models import Generator, MappingNetwork, StyleEncoder
import IPython.display as ipd

speakers = [225,228,229,230,231,233,236,239,240,244,226,227,232,243,254,256,258,259,270,273]

to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4
device = 'cuda:7'
root = '/data/tmpexec/opencode/StarGANv2-VC/'

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema

def compute_style(speaker_dicts, mappingnet, styleencoder):
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        if path == "":
            label = torch.LongTensor([speaker]).to(device)
            latent_dim = mappingnet.shared[0].in_features
            ref = mappingnet(torch.randn(1, latent_dim).to(device), label)
        else:
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = preprocess(wave).to(device)

            with torch.no_grad():
                label = torch.LongTensor([speaker])
                ref = styleencoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)
    
    return reference_embeddings

def main():
    # load F0 model
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(root + "Utils/JDC/bst.t7")['net']
    F0_model.load_state_dict(params)
    _ = F0_model.eval()
    F0_model = F0_model.to(device)
    # load vocoder
    vocoder = load_model(root +  "Vocoder/checkpoint-400000steps.pkl").to(device).eval()
    vocoder.remove_weight_norm()
    _ = vocoder.eval()
    # load starganv2
    model_path = root + 'Models/epoch_00150.pth'
    with open(root + 'Models/config.yml') as f:
        starganv2_config = yaml.safe_load(f)
    starganv2 = build_model(model_params=starganv2_config["model_params"])
    params = torch.load(model_path, map_location='cpu')
    params = params['model_ema']
    _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
    _ = [starganv2[key].eval() for key in starganv2]
    starganv2.style_encoder = starganv2.style_encoder.to(device)
    starganv2.mapping_network = starganv2.mapping_network.to(device)
    starganv2.generator = starganv2.generator.to(device)

    #conversion
    # load input wave
    selected_speakers = [273, 259, 258, 243, 254, 244, 236, 233, 230, 228]
    k = random.choice(selected_speakers)
    src_key = '/p' + str(k)
    wav_path = root + 'Demo/VCTK-corpus/p' + str(k) + '/p' + str(k) + '_023.wav'
    audio, source_sr = librosa.load(wav_path, sr=24000)
    audio = audio / np.max(np.abs(audio))
    audio.dtype = np.float32
    # with reference, using style encoder
    speaker_dicts = {}
    for s in selected_speakers:
        k = s
        speaker_dicts['p' + str(s)] = (root + 'Demo/VCTK-corpus/p' + str(k) + '/p' + str(k) + '_023.wav', speakers.index(s))
    reference_embeddings = compute_style(speaker_dicts, starganv2.mapping_network, starganv2.style_encoder)
    #conversion
    start = time.time()
    
    source = preprocess(audio).to(device)
    keys = []
    converted_samples = {}
    reconstructed_samples = {}
    converted_mels = {}

    for key, (ref, _) in reference_embeddings.items():
        with torch.no_grad():
            f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
            out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)
            
            c = out.transpose(-1, -2).squeeze().to(device)
            y_out = vocoder.inference(c)
            y_out = y_out.view(-1).cpu()

            if key not in speaker_dicts or speaker_dicts[key][0] == "":
                recon = None
            else:
                wave, sr = librosa.load(speaker_dicts[key][0], sr=24000)
                mel = preprocess(wave)
                c = mel.transpose(-1, -2).squeeze().to(device)
                recon = vocoder.inference(c)
                recon = recon.view(-1).cpu().numpy()

        converted_samples[key] = y_out.numpy()
        reconstructed_samples[key] = recon

        converted_mels[key] = out
        
        keys.append(key)
    end = time.time()
    print('total processing time: %.3f sec' % (end - start) )
    #display
    for key, wave in converted_samples.items():
        print('Converted: %s' % key)
        file = ipd.Audio(wave, rate=24000)
        with open(root + 'Demo/VCTK-corpus/' + key + src_key+ '_style.wav', 'wb') as f:
            f.write(file.data)
        print('Reference (vocoder): %s' % key)
        if reconstructed_samples[key] is not None:
            file = ipd.Audio(reconstructed_samples[key], rate=24000)
            with open(root + 'Demo/VCTK-corpus/' + key + src_key+ '_recon.wav', 'wb') as f:
                f.write(file.data)
    """
    print('Original (vocoder):')
    wave, sr = librosa.load(wav_path, sr=24000)
    mel = preprocess(wave)
    c = mel.transpose(-1, -2).squeeze().to(device)
    with torch.no_grad():
        recon = vocoder.inference(c)
        recon = recon.view(-1).cpu().numpy()
    display(ipd.Audio(recon, rate=24000))
    print('Original:')
    display(ipd.Audio(wav_path, rate=24000))
    """


if __name__ == "__main__":
    main()
    #https://github.com/yl4579/StarGANv2-VC