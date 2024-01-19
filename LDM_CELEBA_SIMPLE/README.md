# Latent Diffusion Model
Latent DDPM/DDIM with PyTorch using CelebA Dataset.

### Latent Diffusion Models [LDM]
![image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/1894ddd9-493a-4009-af56-4a974e86cbf3)

## Tips
- I first designed a **Variational AutoEncoder (VAE)** composed of convolutional layers and trained it for <ins>20,000 iterations<ins> with a <ins>batch size of 256<ins>.
- Subsequently, I utilized the trained VAE as the Encoder and Decoder for the **Latent Diffusion Model (LDM)** and trained a U-Net for <ins>50 epochs<ins> with a <ins>batch size of 256<ins>.
- You can find the trained models in the ```weight``` directory, and the dataset can be downloaded through the ```following link```.

|     Celeb-A Faces            |
|:------------------------:|
| [Google Drive](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?resourcekey=0-dYn9z10tMJOBAkviAcfdyQ)   |

## Results
|                       |VAE    |LDM (DDPM Sampling) |
|:--------------:|:-----------:|:-----------:|
| **Img** |![image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/f0556ef5-b57c-4d09-86dc-505889177717)|![image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/2089ed3f-b40e-4502-a343-bb266da755f3)|


 
