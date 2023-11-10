# Localizing anomalies in images

This project ilustrates the Global Average Pooling (GAP) capabilities of replacing fully connected blocks and also enables the Class Activation Map (CAM) to ilustrate localization maps from the classification output.

A simple VGG-16 is used as a CNN classification model. The model output classifier is replaced by a version in which GAP is applied. The fully conected layer is held, but can be removed without major modifications.

## Source

This project is an adaptation to Pytorch from [Dr. Sreenivas Bhattiprolu (Sreeni) GAP example](https://github.com/bnsreenu/python_for_microscopists/blob/master/261_global_average_pooling/261_global_average_pooling.py) and his [Youtube explanation video](https://www.youtube.com/watch?v=gNRVTCf6lvY). 

## Requirements

- Python 3.9.
- Torch 2.1.0 with CUDA 12.1.
