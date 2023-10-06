# CABM: Content-Aware Bit Mapping for Single Image Super-Resolution Network with Large Input (CVPR 2023)

This respository is the official implementation of our CVPR2023 paper.
[paper](https://arxiv.org/abs/2304.06454).


Our implementation is based on [CADyQ(PyTorch)](https://github.com/Cheeun/CADyQ) and [PAMS(PyTorch)](https://github.com/colorjam/PAMS).

Due to the numerous settings in our paper, we have only provided a simplified version of training and testing code here.


### Dependencies
* kornia (pip install kornia)
* Python >= 3.6
* PyTorch >= 1.10.0
* other packages used in our code


### Datasets
* For training, we use [DIV2K datasets](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

* For testing, we use [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) and [Test2K,4K.8K](https://github.com/Cheeun/CADyQ).

```
  # for training
  DIV2K 

  # for testing
  benchmark
  Test2K
  Test4K
```


### How to train CABM step by step
```
# Taking EDSR as an example

# Step-1
# Train full-precision models
sh train_edsrbaseline_org.sh

# Step-2
# Train 8-bit PAMS models
sh train_edsrbaseline_pams.sh

# Step-3
# Train CADyQ models
sh train_edsrbaseline_cadyq.sh

# Step-4
# Get edge-to-bit tables
sh test_edsrbaseline_get_cabm_config.sh

# Step-5
# Get CABM models
sh train_edsrbaseline_cabm_simple.sh
```

### How to test CABM
```
test_edsrbaseline_cabm_simple.sh
```

### How to sampling patches while training
You may refer to [SamplingAUG](https://github.com/littlepure2333/SamplingAug).


### Citation
```
@article{Tian2023CABMCB,
  title={CABM: Content-Aware Bit Mapping for Single Image Super-Resolution Network with Large Input},
  author={Senmao Tian and Ming Lu and Jiaming Liu and Yandong Guo and Yurong Chen and Shunli Zhang},
  journal={ArXiv},
  year={2023},
  volume={abs/2304.06454}
}
```

