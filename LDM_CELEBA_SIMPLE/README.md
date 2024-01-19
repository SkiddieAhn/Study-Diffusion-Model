# Latent Diffusion Model
Latent DDPM/DDIM with PyTorch using CelebA Dataset.

### Latent Diffusion Models [LDM]
![image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/1894ddd9-493a-4009-af56-4a974e86cbf3)

## Description
- I first designed a **Variational AutoEncoder (VAE)** composed of conv layers and trained it for <ins>20,000 iterations with a batch size of 256</ins>.
- I utilized the trained VAE as the Encoder and Decoder for the **Latent Diffusion Model (LDM)** and trained a U-Net for <ins>50 epochs with a <ins>batch size of 256</ins>.
- I performed a denoising process in the latent space, but did not utilize the U-Net as proposed in the latent diffusion paper.
- You can find the trained models in the ```weight``` directory.
- Please download the data from the <ins>Google Drive link</ins> below, unzip the zip file, and then move it to the ```data/celeba``` directory.

|     Celeb-A Faces            |
|:------------------------:|
| [Google Drive](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?resourcekey=0-dYn9z10tMJOBAkviAcfdyQ)   |

## Results
|                       |VAE    |LDM (DDPM Sampling) |LDM (DDIM Sampling) |
|:--------------:|:-----------:|:-----------:|:-----------:|
| **Img** |![image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/f0556ef5-b57c-4d09-86dc-505889177717)|![image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/2089ed3f-b40e-4502-a343-bb266da755f3)| ![image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/e1688f7b-c73f-4393-92e9-2d4c3b2e75eb)|

