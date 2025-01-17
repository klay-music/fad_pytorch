fad_pytorch
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

[Original FAD paper (PDF)](https://arxiv.org/pdf/1812.08466.pdf)

## Install

``` sh
pip install fad_pytorch
```

## About

Features:

- runs in parallel on multiple processors and multiple GPUs (via
  `accelerate`)
- supports multiple embedding methods:
  - VGGish and PANN, both mono @ 16kHz
  - OpenL3 and (LAION-)CLAP, stereo @ 48kHz
- favors ops in PyTorch rather than numpy (or tensorflow)
- `fad_gen` supports WebDataset (audio data stored in S3 buckets)
- runs on CPU, CUDA, or MPS

This is designed to be run as 3 command-line scripts in succession. The
latter 2 (`fad_embed` and `fad_score`) are probably what most people
will want:

1.  `fad_gen`: produces directories of real & fake audio. See
    `fad_gen docs` for calling sequence.
2.  `fad_embed [options] <real_audio_dir> <fake_audio_dir>`: produces
    directories of *embeddings* of real & fake audio
3.  `fad_score [optoions] <real_emb_dir> <fake_emb_dir>`: reads the
    embeddings & generates FAD score, for real (“$r$”) and fake (“$f$”):

$$ FAD = || \mu_r - \mu_f ||^2 + tr\left(\Sigma_r + \Sigma_f - 2 \sqrt{\Sigma_r \Sigma_f}\right)$$

## Comments / FAQ / Troubleshooting

- “`RuntimeError: CUDA error: invalid device ordinal`”: This happens
  when you have a “bad node” on an AWS cluster. [Haven’t yet figured out
  what causes it or how to fix
  it](https://discuss.huggingface.co/t/solved-accelerate-accelerator-cuda-error-invalid-device-ordinal/21509/1).
  Workaround: Just add the current node to your SLURM `--exclude` list,
  exit and retry. Note: it may take as many as 5 to 7 retries before you
  get a “good node”.
- “FAD scores obtained from different embedding methods are *wildly*
  different!” …Yea. It’s not obvious that scores from different
  embedding methods should be comparable. Rather, compare different
  groups of audio files using the same embedding method, and/or check
  that FAD scores go *down* as similarity improves.
- “FAD score for the same dataset repeated (twice) is not exactly zero!”
  …Yea. There seems to be an uncertainty of around +/- 0.008. I’d say,
  don’t quote any numbers past the first decimal point.

## Contributing

This repo is still fairly “bare bones” and will benefit from more
documentation and features as time goes on. Note that it is written
using [nbdev](https://nbdev.fast.ai/), so the things to are:

1.  fork this repo
2.  clone your fork to your (local) machine
3.  Install nbdev: `python3 -m pip install -U nbdev`
4.  Make changes by editing the notebooks in `nbs/`, not the `.py` files
    in `fad_pytorch/`.
5.  Run `nbdev_export` to export notebook changes to `.py` files
6.  For good measure, run `nbdev_install_hooks` and `nbdev_clean` -
    especially if you’ve *added* any notebooks.
7.  Do a `git status` to see all the `.ipynb` and `.py` files that need
    to be added & committed
8.  `git add` those files and then `git commit`, and then `git push`
9.  Take a look in your GitHub Actions tab, and see if the “test” and
    “deploy” CI runs finish properly (green light) or fail (red light)
10. One you get green lights, send in a Pull Request!

*Feel free to ask me for tips with nbdev, it has quite a learning curve.
You can also ask on [fast.ai forums](https://forums.fast.ai/) and/or
[fast.ai
Discord](https://discord.com/channels/689892369998676007/887694559952400424)*

## Related Repos

There are \[several\] others, but this one is mine. These repos didn’t
have all the features I wanted, but I used them for inspiration:

- https://github.com/gudgud96/frechet-audio-distance
- https://github.com/google-research/google-research/tree/master/frechet_audio_distance:
  Goes with [Original FAD paper](https://arxiv.org/pdf/1812.08466.pdf)
- https://github.com/AndreevP/speech_distances
