# Facial Foundational Model Advances Early Warning of Coronary Artery Disease from Live Videos with DigitalShadow



## Installation (minimal)

```bash
sudo apt install curl
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
mamba create -n ds python=3.10 -y
mamba activate ds
mamba install pytorch==2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python==4.11.0.86 face_recognition==1.3.0 supervision==0.25.1 timm==0.4.12 tensorboardX==2.6.2.2 numpy==1.26.4
```



## Get Started

Run this command to start a simple example.

`python app.py`



## Citation

Our paper has been accepted by **XXX**.

If you find DigitalShadow to be helpful in your research or applications, please cite it using this BibTeX:

```
XXX
```
