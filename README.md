# Generative Modelling Course Project
## HW 1: Language Model
### Environment
Two versions of training source code are provided. `train.ipynb` runs in a Google Colab environment, while `train.py`, as well as the evaluation code, `evaluate.py`, runs in a Conda environment created as so:
```bash
conda create -n genmod_lang python=3.11
conda activate genmod_lang
pip install transformers datasets evaluate
pip install accelerate -U
```

### How to Run
#### Training
For `train.ipynb`, simply run it in a Google Colab environment!

As for `train.py`, after you've set up the conda environment specified above, simply run it with
```bash
python3 codes/train.py
```

A note: please run the code with the **main directory** (i.e., the one containing `ckpts`, `codes`, etc.) **as the current working directory**. Otherwise, the paths gets confused!

#### Evaluate
Change the prompt as you desire here:
```python
prompt = "Two households, both alike in dignity,"	# Set your prompt here
```

And run it with:
```bash
python3 codes/evaluate.py
```

### Technical Report
For the first homework, I chose to write the code myself, with the Huggingface `transformers` and `datasets` libraries. The bulk of the workload comes from getting to know how to use these libraries.

#### Code Implementation

The training code is fairly straightforward. I utilized methods and classes provided by the `transformers` and `datasets` libraries to download the dataset, tokenize and collate it, train it on the training dataset, and evaluate its perplexity.

I chose the `OPT` model by Meta AI and specifically its 350M varient as `OPT` is reported to be the best performing model on the WikiText2 dataset, and I chose the 350M varient as it provides a nice balance between performance and ease of training.

A great part of the workload came from choosing the correct class for the model (I originally used the `OPTModel` class), and learning what tokenizer and collator to pass in through trial and error.

#### Training

The training was carried out in a Google Colab runtime with a single V100 GPU and 12GB of system memory (at the time of training, A100s were not available).

The V100 GPU, with only 16GB of VRAM, presented a challenge during training. I set the batch size to 2 and enabled 4-step gradient accumulation to fit the dataset into the video memory.

After training for 1 epoch, the evaluation code returned a perplexity of 15.97, which was reasonable given that `OPT-175B` achieved a perplexity of 8.34 in the same task.

## HW 2: Image Model
### Environment
Run the following instructions in the shell to set up the environment:
```bash
conda create -n genmod_img python=3.11
conda activate genmod_img
pip install datasets diffusers transformers torch torchvision accelerate
```

### How to Run
`cd` into the main directory. Then run
```bash
python3 codes/train.py
```
to train, and run
```bash
python3 codes/inference.py
```
to generate new images.

### Technical Report
#### Code

I chose to use the Huggingface's `diffusers`, `accelerate` and `datasets` to implement my training and inference code. This code is heavily based on one of Huggingface's tutorials: https://huggingface.co/docs/diffusers/tutorials/basic_training.

To make the model more robust, we've implemented random mirroring to the images.

As for the architecture of the model, I followed a standard architecture for the `UNet2DModel`.

I used the `AdamW` optimizer, and also employed learning rate warmup, for a more stable training and faster convergence.

#### Training

The `TrainingConfig` was tuned to make full use of the GPU's vRAM and speed up training as much as possible. It is for this purpose that I've set the training and evaluation batch size to 128, chose a gradient accumulation step count of 4, and enabled `fp16` mixed precision, the latter 2 of these choices proven to have little impact on the quality of the generated images.

On a single V100 GPU in a Google Colab runtime, 50 epochs were finished in about 80 minutes with the loss of the last epoch at 0.0532 (but note that at this point, the loss is fluctuating quite a bit between epochs).

Inspecting the evaluation images, we can see that most of them have a well-defined main object, but their shape are often somewhat bizzare. This might be due to the low resolution of the CIFAR-10 dataset, along with the unconditional training method I've adopted, in which every image in the dataset is fed into the model indiscriminately.

## HW 3: Diffusion Acceleration with DiffuseVAE and Quantization
In this project, we replicated the results of the DiffuseVAE paper, and attempted to make improvements to the inference speed and generation quality.

See our paper at: `final/Generative_Model.pdf`

See our cherrypicked replication results at: `final/DiffuseVAE/cherrypicked`

See our full replication results, including metrics and generated images, at: `final/DiffuseVAE/replication`
