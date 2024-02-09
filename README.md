# Latent Feature Classification in Non-intrusive Speech Quality Assessment

Authors: Fredrik Cumlin (fcumlin@gmail.com), Xinyu Liang (hopeliang990504@gmail.com)

This is the official implementation of latent distortion classification in non-intrusive speech quality assessment from the paper "Latent of DNN-based Speech Quality Assessment Method Classifies Distortions".

## Download datasets

### LibriAugmented
The authors are unable due to license restrictions to make the dataset open access. To create you're own dataset, a description on expected folder and csv structure can be found in `dataset.LibriAugmented`.

### ESC-50 dataset
See https://github.com/karolpiczak/ESC-50.

## Training and evaluation
We have made the code base [gin configurable](https://github.com/google/gin-config), which is a light-weight configuration framework. See `configs/tot.gin` for an example. Example run:
```
python train.py --gin_path=configs/tot.gin --save_dir=path/to/save_dir
```

For the users convinience, we have provided with an explicit kNN evaluation script on the ESC-50 dataset, `esc50_eval.py`. To make it more general, we have defined an 'abstract' function which should be updated with appropriate model call. Example run:
```
python esc50_eval.py
```

## Acknowledgement
We would like to thank our co-authors of the paper for great insights and discussions (without consideration of order): Victor Ungureanu (ungureanu@google.com), Chandan K.A. Reddy (chandanka@google.com), Christian Schüldt (cschuldt@google.com), and Saikat Chatterjee (sach@kth.se).

We also would like to thank the Knut and Alice Wallenberg Foundation for providing computation resources at the Berzelius cluster at the National Supercomputer Centre.
