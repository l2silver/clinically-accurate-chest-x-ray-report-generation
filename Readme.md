## Clinically Accurate Chest X-Ray Report Generation

PyTorch Implementation of *Clinically Accurate Chest X-Ray Report Generation* [Paper](https://arxiv.org/abs/1904.02633)<br/>

Implementation based on the following repos:

1) https://github.com/fawazsammani/knowing-when-to-look-adaptive-attention
2) https://github.com/ZexinYan/Medical-Report-Generation
3) https://github.com/ZexinYan/im2p-pytorch

## To Do

- [ ] Add pretrained embedder to word decoder
- [x] Add loss functions
- [x] word encoder should output a vector representing hot encoding of word
- [ ] reinforcement learning section
- [ ] add validation section
- [ ] add testing section

## Thoughts

* in training, should loss function look at embedding instead of hot encoding for error?

## Acknowledgements

This repo would not have been possible with the help and support of [Alex Dela Cruz](https://www.linkedin.com/in/alex-dela-cruz-89730175)
