## 4DenoiseNet: Adverse Weather Denoising from Adjacent Point Clouds, [arXiv](https://arxiv.org/abs/2209.07121)

![](https://github.com/alvariseppanen/4DenoiseNet/blob/main/demo.gif)

### Download SnowyKITTI-dataset:

coming soon

### Train:
```
cd networks
./train.sh -d root/snowyKITTI/dataset/ -a fourdenoisenet.yml -l /your/log/folder/ -c 0
```

### Infer (pretrained model -m root/logs/2022-9-01-10:40/):
```
cd networks/train/tasks/semantic
python3 infer.py -d root/toy_snowyKITTI/dataset/ -m root/logs/2022-9-01-10:40/ -l /your/predictions/folder/ -s test
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
python3 snow_visualize.py -d root/toy_snowyKITTI/dataset/ -dc root/networks/train/tasks/semantic/config/labels/snowy-kitti.yaml -p /your/predictions/folder/ -s 22
(-s = sequence)
```

Thanks to [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext) for providing some of the code! 
