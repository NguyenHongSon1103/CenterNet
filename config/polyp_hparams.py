hparams = dict(
    num_classes = 2,
    names = ["adenomatous", "hyperplastic"],
    #----------------------- 
    # Model architecture
    #-----------------------
    backbone = 'resnet101',
    neck = 'fpn_asf',
    head = 'centernet',
    #-----------------------
    # Only config for cspdarknet backbone
    # One of ['tiny', 's', 'm', 'l', 'x']
    #-----------------------
    yolox_phi = 'm',   
    weight_decay = 5e-4,
    #-----------------------
    # Other configs
    #-----------------------
    input_size = 512,
    max_objects = 100,
    score_threshold = 0.2,
    model_dir = 'trained_models/20221005',
    batch_size=3, 
    epochs = 100,
    resume = False,
    pretrained_weights = '',
    optimizer = dict(
        type = 'adam',
        base_lr = 5e-4,
        end_lr = 1e-5,
        warmup_lr = 2e-5,
        warmup_steps = 10000
    ),
    fine_tune_checkpoint = False,
    misc_effect=None,
    visual_effect=None,
    label_map = 'config/classes_polyp.json',
)