hparams = dict(
    num_classes = 4,
    backbone_type = 'resnet50',
    input_size = 512,
    max_objects = 10,
    score_threshold = 0.2,
    model_dir = 'trained_models/dummy_resnet',
    batch_size=1, 
    epochs = 1000,
    resume = False,
    pretrained_weights = '',
    optimizer = dict(
        type = 'adam',
        base_lr = 1e-3,
        end_lr = 1e-5,
    ),
    fine_tune_checkpoint = False,
    misc_effect=None,
    visual_effect=None,
    label_map = 'config/classes_dummy.json',
    data_path = ''
)