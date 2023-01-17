## 4DenoiseNet: Adverse Weather Denoising from Adjacent Point Clouds, [Publication](https://ieeexplore.ieee.org/document/9976208)

![](https://github.com/alvariseppanen/4DenoiseNet/blob/main/demo.gif)


### Citation:

@ARTICLE{9976208,
  author={Seppanen, Alvari and Ojala, Risto and Tammi, Kari},
  journal={IEEE Robotics and Automation Letters}, 
  title={4DenoiseNet: Adverse Weather Denoising from Adjacent Point Clouds}, 
  year={2022},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/LRA.2022.3227863}}


### SnowyKITTI-dataset:

[Download](https://www.dropbox.com/s/o3r654cdzfl405d/snowyKITTI.zip?dl=0)


### Train:
```
cd networks
./train.sh -d root/snowyKITTI/dataset/ -a fourdenoisenet.yml -l /your/log/folder/ -c 0
```

### Infer (pretrained model -m root/logs/2023-1-17-08:49/):
```
cd networks/train/tasks/semantic
python3 infer.py -d root/toy_snowyKITTI/dataset/ -m root/logs/2023-1-17-08:49/ -l /your/predictions/folder/ -s test
(-s = split)
```

### Evaluate:
```
cd networks/train/tasks/semantic
python3 snow_evaluate_iou.py -d root/toy_snowyKITTI/dataset/ -dc root/networks/train/tasks/semantic/config/labels/snowy-kitti.yaml -p /your/predictions/folder/ -s test
(-s = split)
```

### Visualize:
```
cd utils
python3 snow_visualize.py -d root/toy_snowyKITTI/dataset/ -c root/networks/train/tasks/semantic/config/labels/snowy-kitti.yaml -p /your/predictions/folder/ -s 22
(-s = sequence)
```

Thanks to [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext) for providing some of the code! 
