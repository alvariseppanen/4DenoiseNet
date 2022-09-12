## The official implementation of 4DenoiseNet


## Train:
'''
cd networks
./train.sh -d root_to_snowyKITTI/snowyKITTI/dataset/ -a fourdenoisenet.yml -l /your/log/folder/ -c 0
'''

## Infer with pretrained model:
cd networks/train/tasks/semantic
python3 infer.py -d root_to_snowyKITTI/snowyKITTI/dataset/ -m root/logs/2022-9-01-10:40/ -l /your/predictions/folder/ -s test
(-s = split)

## Evaluate:
cd networks/train/tasks/semantic
python3 snow_evaluate_iou.py -d root_to_snowyKITTI/snowyKITTI/dataset/ -dc root/networks/train/tasks/semantic/config/labels/snowy-kitti.yaml -p /your/predictions/folder/ -s test
(-s = split)

## Visualize:
python3 snow_visualize.py -d root_to_snowyKITTI/snowyKITTI/dataset/ -dc root/networks/train/tasks/semantic/config/labels/snowy-kitti.yaml -p /your/predictions/folder/ -s 11
(-s = sequence)

Thanks to SalsaNext for providing some of the code! https://github.com/TiagoCortinhal/SalsaNext
