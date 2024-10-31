# Generative Models HW 2
## Environment
Run the following instructions in the shell to set up the environment:
```bash
conda create -n genmod_img python=3.11
conda activate genmod_img
pip install datasets diffusers transformers torch torchvision accelerate
```

## How to Run
`cd` into the main directory. Then run
```bash
python3 codes/train.py
```
to train, and run
```bash
python3 codes/inference.py
```
to generate new images.