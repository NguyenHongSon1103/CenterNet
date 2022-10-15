hparams = dict(
    num_classes = 1,
    names = ['polyps'],
    name2idx = {"adenomatous":0, "hyperplastic":0},
    #names = ["adenomatous", "hyperplastic"],
    #----------------------- 
    # Model architecture
    #-----------------------
    backbone = 'cspdarknet',
    neck = '',
    #head = 'centernet',
    head = 'ihead',
    #-----------------------
    # Only config for cspdarknet backbone
    # One of ['tiny', 's', 'm', 'l', 'x']
    #-----------------------
    yolox_phi = 's',   
    weight_decay = 5e-4,
    #-----------------------
    # Other configs
    #-----------------------
    input_size = 512,
    max_objects = 100,
    score_threshold = 0.2,
    model_dir = 'trained_models/20221012',
    batch_size=8, 
    epochs = 100,
    resume = False,
    pretrained_weights = '',
    optimizer = dict(
        type = 'adam',
        base_lr = 1e-3,
        end_lr = 1e-5,
        warmup_lr = 2e-5,
        warmup_steps = 10000
    ),
    fine_tune_checkpoint = False,
    misc_effect=None,
    visual_effect=None,
    label_map = 'config/classes_polyp.json',
)
