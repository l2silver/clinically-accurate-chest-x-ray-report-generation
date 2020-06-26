## Clinically Accurate Chest X-Ray Report Generation

PyTorch Implementation of *Clinically Accurate Chest X-Ray Report Generation* [Paper](https://arxiv.org/abs/1904.02633)<br/>

Implementation based on the following repos:

1) https://github.com/fawazsammani/knowing-when-to-look-adaptive-attention
2) https://github.com/ZexinYan/Medical-Report-Generation
3) https://github.com/ZexinYan/im2p-pytorch

## Getting Started

1) Install dependencies using conda environment file env.yml

2) Download images and reports from https://openi.nlm.nih.gov/faq

3) Edit in preprocessing.py the following line to point to your data:
```
ImagesReports('./data/nlm/images', './data/nlm/reports', device=device, transform=preprocess)
```
4) Train: python training.py


## To Do

- [ ] Add pretrained embedder to word decoder
- [x] Add loss functions
- [x] word encoder should output a vector representing hot encoding of word
- [ ] reinforcement learning section
- [ ] add validation section
- [ ] add testing section
- [ ] save model every epoch
- [ ] change to receive arguments from commandline

## Thoughts

* in training, should loss function look at embedding instead of hot encoding for error?

## Acknowledgements

This repo would not have been possible with the help and support of [Alex Dela Cruz](https://www.linkedin.com/in/alex-dela-cruz-89730175)
