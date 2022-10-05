hparams = dict(
    num_classes = 20,
    backbone_type = 'resnet50',
    input_size = 512,
    max_objects = 100,
    score_threshold = 0.2,
    model_dir = 'trained_models/voc_20220629',
    batch_size=4,
    epochs = 50,
    resume = False,
    pretrained_weights = '',
    optimizer = dict(
        type = 'adam',
        base_lr = 1e-3,
        end_lr = 1e-5,
    ),
    fine_tune_checkpoint = False,
    multi_scale=False,
    multi_image_sizes=(320, 352, 384, 416, 448, 480, 512, 544, 576, 608),
    misc_effect=None,
    visual_effect=None,
    label_map = 'config/classes_voc.json',
    data_path = ''
)