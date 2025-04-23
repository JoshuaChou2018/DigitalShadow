# Facial Foundational Model Advances Early Warning of Coronary Artery Disease from Live Videos with DigitalShadow



## Quick Installation (Inference-Only)

To get started with **DigitalShadow** for inference, run the following minimal setup:

```bash
# Install Mambaforge
sudo apt install curl
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh

# Create and activate the environment
mamba create -n ds python=3.10 -y
mamba activate ds

# Install required packages
mamba install pytorch==2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install cmake==4.0.0
pip install opencv-python==4.11.0.86 face_recognition==1.3.0 supervision==0.25.1 timm==0.4.12 tensorboardX==2.6.2.2 numpy==1.26.4
```



## Model Access

- To request access to the **CAD risk prediction model**, please contact us at **xin.gao@kaust.edu.sa**.



## Get Started

Before running the prediction, ensure the CAD risk prediction model is placed inside the `model` directory.



To run a basic example:

`python app.py`



To process other videos:

`python app.py --video path/to/video`





## Citation

If you find **DigitalShadow** valuable for your research or applications, please cite us:

```
N/A
```
