# Generative Models Homework 2 - Report

## Code

I chose to use the Huggingface's `diffusers`, `accelerate` and `datasets` to implement my training and inference code. This code is heavily based on one of Huggingface's tutorials: https://huggingface.co/docs/diffusers/tutorials/basic_training.

To make the model more robust, we've implemented random mirroring to the images.

As for the architecture of the model, I followed a standard architecture for the `UNet2DModel`.

I used the `AdamW` optimizer, and also employed learning rate warmup, for a more stable training and faster convergence.



## Training

The `TrainingConfig` was tuned to make full use of the GPU's vRAM and speed up training as much as possible. It is for this purpose that I've set the training and evaluation batch size to 128, chose a gradient accumulation step count of 4, and enabled `fp16` mixed precision, the latter 2 of these choices proven to have little impact on the quality of the generated images.

On a single V100 GPU in a Google Colab runtime, 50 epochs were finished in about 80 minutes with the loss of the last epoch at 0.0532 (but note that at this point, the loss is fluctuating quite a bit between epochs).

Inspecting the evaluation images, we can see that most of them have a well-defined main object, but their shape are often somewhat bizzare. This might be due to the low resolution of the CIFAR-10 dataset, along with the unconditional training method I've adopted, in which every image in the dataset is fed into the model indiscriminately.