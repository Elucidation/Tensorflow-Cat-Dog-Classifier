Tensorflow Cats vs Dogs classifier
---

We build our own dataset from existing flicker images of cats and dogs, and then train a tensorflow neural network to classify cats and dogs.

Run using `./cats_vs_dogs.py`

Currently a single layer NN, no successful learning yet

```
usage: cats_vs_dogs.py [-h] [--cat_dir CAT_DIR] [--dog_dir DOG_DIR]
                       [--num_steps NUM_STEPS]

optional arguments:
  -h, --help            show this help message and exit
  --cat_dir CAT_DIR     Directory for storing input cat images
  --dog_dir DOG_DIR     Directory for storing input dog images
  --num_steps NUM_STEPS
                        Number of steps to train model
```

# Building a dataset

`resize_images.py` contains a script to resize all passed in images into
64x64 grayscale pngs named `####.png` monotonically increasing in the 
specified output folder

```
Usage: resize_images.py [options] image1 [image2 ...]

Options:
  -h, --help            show this help message and exit
  -o OUTPUT_FOLDER, --output_folder=OUTPUT_FOLDER
                        Output folder to save resized images to
  -n MAX_N, --max_number=MAX_N
                        Maximum number of images to process
  -d, --dryrun          Do a dry run (no processing/saving)
```