## 4DenoiseNet: Adverse Weather Denoising from Adjacent Point Clouds, [arXiv](https://arxiv.org/abs/2209.07121)

![](https://github.com/alvariseppanen/4DenoiseNet/blob/main/demo.gif)


### Citation:
Coming soon, accepted by IEEE Robotics and Automation Letters


### SnowyKITTI-dataset:

[Download](https://www.dropbox.com/s/o3r654cdzfl405d/snowyKITTI.zip?dl=0)


### Train:
```
cd networks
./train.sh -d root/snowyKITTI/dataset/ -a fourdenoisenet.yml -l /your/log/folder/ -c 0
```

### Infer (pretrained model -m root/logs/2022-9-22-20:56/):
```
cd networks/train/tasks/semantic
python3 infer.py -d root/toy_snowyKITTI/dataset/ -m root/logs/2022-9-22-20:56/ -l /your/predictions/folder/ -s test
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
