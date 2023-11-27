# Localizing anomalies in images

This project ilustrates the Global Average Pooling (GAP) capabilities of replacing fully connected blocks and also enables the Class Activation Map (CAM) to ilustrate localization maps from the classification output.

A simple VGG-16 is used as a CNN classification model. The model output classifier is replaced by a version in which GAP is applied. The GAP block allows the extraction of CAM. 

## Source

This project is an adaptation to Pytorch from [Dr. Sreenivas Bhattiprolu (Sreeni) GAP example](https://github.com/bnsreenu/python_for_microscopists/blob/master/261_global_average_pooling/261_global_average_pooling.py) and his [Youtube explanation video](https://www.youtube.com/watch?v=gNRVTCf6lvY).

## Requirements

- Python 3.9.
- Torch 2.1.0 with CUDA 12.1.

## Train and validation

The weights are made available in this repository. To evaluate it, just download it put it in the root folder with name 'best_model.pth' and run the model in inference mode.  

The model can be used as follows:

```python
CUDA_VISIBLE_DEVICES=0 python main.py --image_dir $your_dir --mode [train|eval] --visualize false --num_epochs 10 --batch_size 32 --learning_rate 1e-3 --image_size 224
```

Note that visualize param will allow the last batch images and CAM's to be saved in a file (only in eval mode).

## Data

- The dataset consists of cell images infected or not with Malaria. The data can be obtained in [National Library of Medicine](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets). 80%|20% was the policy for the train/validation splits creation. There are two classes: infected and uninfected. A sample of both classes are shown below.

- Uninfected example.

![Caption for subfigure (a).](results/img_5.png)

- Infected/parasitized example.

![Caption for subfigure (b).](results/img_31.png)



## Results

The best results achieved by the classifier was:
validation loss of 0.511 and validation accuracy of 0.860.

### CAM visualization

The more hot colour in the CAM image, the more probability that the pixels are in a region of interest (parasitized area) in the model inference. The more cold colour, the more probability that the region is a non infected area.

![Sample image (a).](results/img_14.png)

![CAM for sample image (a).](results/cam_14.png)

![Sample image (b).](results/img_31.png)

![CAM for sample image (b).](results/cam_31.png)

### Loss and quantitative results

- Loss:

![Train/validation Loss.](results/loss.png)

- ROC curve:

![Train/validation ROC curve.](results/roc_curve.png)

- Confusion matrix:

![Train/validation confusion matrix.](results/output.png)