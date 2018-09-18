## Learning Dynamic Memory Networks for Object Tracking
### Introduction
This is the Tensorflow implementation of our [**MemTrack**](https://arxiv.org/pdf/1803.07268.pdf) tracker published in ECCV, 2018. Detailed comparision results can be found in the author's [webpage](https://tianyu-yang.com)

### Prerequisites

* Python 3.5 or higher
* Tensorflow 1.2.1 or higher
* CUDA 8.0

### Path setting
Set proper `home_path` in `config.py` accordingly in order to proceed the following step. Make sure that you place the tracking data properly according to your path setting.

### Tracking Demo
You can use our pretrained model to test our tracker first. 
1. Download the model from the link: [GoogleDrive](https://drive.google.com/open?id=1ybywFIzVbflj2-n3gDM9kqEP2-LZ4f3l)
2. Put the model into directory `./output/models`
3. Run `python3 demo.py` in directory `./tracking`

### Training
1. Download the ILSRVC data from the official website and extract it to proper place according to the path in `config.py`.
2. Then run the `sh process_data.sh` in `./build_tfrecords` directory to convert ILSVRC data to tfrecords.
3. Run `python3 experiment.py` to train the model.

### Citing MemTrack
If you find the code is helpful, please cite
```
@inproceedings{Yang2018,
	author = {Yang, Tianyu and Chan, Antoni B.},
	booktitle = {ECCV},
	title = {{Learning Dynamic Memory Networks for Object
	Tracking}},
	year = {2018}
}
```
