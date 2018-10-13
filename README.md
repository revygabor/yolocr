# YOLOCR

## Purpose and conception

This repository is our homework for the Practical deep learning course at Budapest University of Technology and Economics.

Our main goal is to create a optical character recognition based on YOLO ([v1][], [v2][], [v3][]) architecture. We would like to achieve the best possible speed and accuracy by refining YOLO.

[v1]: https://arxiv.org/abs/1506.02640v5
[v2]: https://arxiv.org/abs/1612.08242v1
[v3]: https://arxiv.org/abs/1804.02767v1

## Data for training

We use [SynthText](https://github.com/ankush-me/SynthText/tree/df18cd1c0969bdbd0890cb6b9700d96caedfa943) and its pregenerated dataset, which includes approximately 800 thousands synthetic scene-text images and annotations with word-level and character-level bounding-boxes. However, these bounding-boxes are given as simple quadrilaterals and needs converting to rectangles. 

### Bounding-box transformation

We want to train our model for recognising rectangle bounding-boxes, and in order to do that we have to convert the bounding-boxes given by SynthText.
We take the longest side of the given quadrilateral and it will be overlapped by our rectangle's longer side.
After that we make sure that all points of the original bounding-box will be inside our new bounding-rectangle.