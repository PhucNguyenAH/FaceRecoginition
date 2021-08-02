# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 2,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

cfg_efb0 = {
    'name': 'Efficientnet-b0',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 301,
    'decay1': 300,
    'decay2': 300,
    'image_size': 840,
    'pretrain': False,
    'return_layers': {'blocks1': 1, 'blocks2': 2, 'blocks3': 3},
    'in_channel': 20,
    'out_channel': 256
}

cfg_efb1 = {
    'name': 'Efficientnet-b1',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 3,
    'ngpu': 1,
    'epoch': 171,
    'decay1': 170,
    'decay2': 170,
    'image_size': 840,
    'pretrain': False,
    'return_layers': {'blocks1': 1, 'blocks2': 2, 'blocks3': 3},
    'in_channel': 20,
    'out_channel': 256
}

cfg_efb2 = {
    'name': 'Efficientnet-b2',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 2,
    'ngpu': 1,
    'epoch': 61,
    'decay1': 60,
    'decay2': 60,
    'image_size': 840,
    'pretrain': False,
    'return_layers': {'blocks1': 1, 'blocks2': 2, 'blocks3': 3},
    'in_channel': 24,
    'out_channel': 256
}

cfg_efb3 = {
    'name': 'Efficientnet-b3',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 2,
    'ngpu': 1,
    'epoch': 101,
    'decay1': 100,
    'decay2': 100,
    'image_size': 840,
    'pretrain': False,
    'return_layers': {'blocks1': 1, 'blocks2': 2, 'blocks3': 3},
    'in_channel': 24,
    'out_channel': 256
}

cfg_efb4 = {
    'name': 'Efficientnet-b4',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 2,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 250,
    'decay2': 250,
    'image_size': 840,
    'pretrain': False,
    'return_layers': {'blocks1': 1, 'blocks2': 2, 'blocks3': 3},
    'in_channel': 20,
    'out_channel': 256
}

