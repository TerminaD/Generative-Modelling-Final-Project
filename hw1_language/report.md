# Generative Models HW 1 - Report

For the first homework, I chose to write the code myself, with the Huggingface `transformers` and `datasets` libraries. The bulk of the workload comes from getting to know how to use these libraries.

## Code Implementation

The training code is fairly straightforward. I utilized methods and classes provided by the `transformers` and `datasets` libraries to download the dataset, tokenize and collate it, train it on the training dataset, and evaluate its perplexity.

I chose the `OPT` model by Meta AI and specifically its 350M varient as `OPT` is reported to be the best performing model on the WikiText2 dataset, and I chose the 350M varient as it provides a nice balance between performance and ease of training.

A great part of the workload came from choosing the correct class for the model (I originally used the `OPTModel` class), and learning what tokenizer and collator to pass in through trial and error.

## Training

The training was carried out in a Google Colab runtime with a single V100 GPU and 12GB of system memory (at the time of training, A100s were not available).

The V100 GPU, with only 16GB of VRAM, presented a challenge during training. I set the batch size to 2 and enabled 4-step gradient accumulation to fit the dataset into the video memory.

After training for 1 epoch, the evaluation code returned a perplexity of 15.97, which was reasonable given that `OPT-175B` achieved a perplexity of 8.34 in the same task.

