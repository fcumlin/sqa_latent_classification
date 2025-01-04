# Impairment classification using latent features in non-intrusive speech quality models

Authors: Fredrik Cumlin (fcumlin@gmail.com), Xinyu Liang (hopeliang990504@gmail.com)

This is the official implementation of latent distortion classification in non-intrusive speech quality assessment from the paper "Impairments are Clustered in Latents of Deep Neural Network-based Speech Quality Models".

## Download datasets

### LibriAugmented
The authors are unable to make the dataset open access due to licensing restrictions. Note that we have used [LibriSpeech](https://www.openslr.org/94/) as a clean speech dataset, and distorted the signals using augmentations from [audiomentations](https://github.com/iver56/audiomentations).

### ESC-50 dataset
See https://github.com/karolpiczak/ESC-50.

## Training and evaluation
We have made the codebase [gin configurable](https://github.com/google/gin-config), which is a lightweight configuration framework. See `configs/tot.gin` for an example. Example run:
```
python train.py --gin_path=configs/tot.gin --save_dir=path/to/save_dir
```

## Acknowledgement
We would like to thank our co-authors of the paper for great insights and discussions (without consideration of order): Victor Ungureanu (ungureanu@google.com), Chandan K.A. Reddy (chandanka@google.com), Christian Schüldt (cschuldt@google.com), and Saikat Chatterjee (sach@kth.se).

We also would like to thank the Knut and Alice Wallenberg Foundation for providing computational resources at the Berzelius cluster at the National Supercomputer Centre.

## License
MIT License

Copyright (c) 2025 Fredrik Cumlin, Royal Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
