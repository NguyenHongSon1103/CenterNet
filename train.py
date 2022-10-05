import tensorflow as tf
from losses import loss
import time
from config.polyp_hparams import hparams
import datetime
import os
import json
from generators.dataset_custom import get_polyp_generator
from models.model import Model
import numpy as np
import math
from tqdm import tqdm
from eval.eval_callback import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

root = os.getcwd()
save_model_dir = os.path.join(root, hparams['model_dir'])
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)
## Get dataset ##
train_gen, val_gen = get_polyp_generator(hparams)
# train_gen, val_gen = get_voc_generator(hparams)
print('DATASET: \n TRAINING DATA: %d \n VALIDATION DATA: %d'%(len(train_gen), len(val_gen)))

## Define model ##
input_shape = (hparams['input_size'], hparams['input_size'], 3)
model = Model(hparams['backbone'], hparams['neck'], hparams['head'],
             input_shape, hparams['num_classes'], hparams['yolox_phi'], hparams['weight_decay']).build()

model.summary()
with open(os.path.join(save_model_dir, 'model.json'), 'w') as f:
    f.write(model.to_json())
if hparams['resume']:
    model.load_weights(hparams['pretrained_weights'])

# lr = hparams['optimizer']['base_lr']

class CosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = 0.0

    def __call__(self, step):
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.initial_learning_rate * decayed  
    
class Warmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_lr, target_lr, warmup_steps):
        self.warmup_lr = warmup_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        lr = self.warmup_lr + step*(self.target_lr - self.warmup_lr)/self.warmup_steps
        return lr 
    
# lr = CosineDecay(
#     hparams['optimizer']['base_lr'], hparams['epochs']*len(train_gen))
# lr = tf.keras.optimizers.schedules.CosineDecay(
#     hparams['optimizer']['base_lr'], hparams['epochs']*len(train_gen), alpha=0.0)
# lr = 1e-4
warmup_lr = Warmup(
    hparams['optimizer']['warmup_lr'],
    hparams['optimizer']['base_lr'],
    warmup_steps = hparams['optimizer']['warmup_steps'])

if hparams['optimizer']['type'] == 'adam':
    warmup_opt = tf.keras.optimizers.Adam(learning_rate = warmup_lr)
else:
    warmup_opt = tf.keras.optimizers.SGD(learning_rate = warmup_lr)

lr = tf.keras.optimizers.schedules.PolynomialDecay(
    hparams['optimizer']['base_lr'], 300000,
    hparams['optimizer']['end_lr'], power=1.0,)

if hparams['optimizer']['type'] == 'adam':
    opt = tf.keras.optimizers.Adam(learning_rate = lr)
else:
    opt = tf.keras.optimizers.SGD(learning_rate = lr)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join(save_model_dir, 'logs', 'train')
val_log_dir = os.path.join(save_model_dir, 'logs', 'val')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

init = np.zeros((hparams['epochs']+1))
train_info = {'total_loss':init.copy(), 'hm_loss': init.copy(),
              'wh_loss': init.copy(),
              'reg_loss':init.copy()}

val_info = {'total_loss':init.copy(), 'hm_loss': init.copy(),
              'wh_loss': init.copy(),
              'reg_loss':init.copy()}


#----------------------
# Save hyper-parameters
#----------------------
with open(os.path.join(save_model_dir, 'hparams.json'), 'w') as f:
    f.write(json.dumps(hparams))
best_mAP = 0
resume_epoch = 1
if hparams['resume']:
    try:
        resume_epoch = int(hparams['pretrained_weights'].split('_')[1])
        best_mAP = float(hparams['pretrained_weights'].split('_')[-1][:-3])
    except:
        pass
for epoch in range(resume_epoch, hparams['epochs']+1):
    epoch_total_avg = tf.keras.metrics.Mean() # Keeping track of the training loss
    epoch_hm_avg = tf.keras.metrics.Mean() # Keeping track of the training accuracy
    epoch_wh_avg = tf.keras.metrics.Mean()
    epoch_reg_avg = tf.keras.metrics.Mean()
    
    print('-'*20, 'Epoch %d:'%epoch, '-'*20)
    print('%10s\t%10s\t%10s\t%10s'%('hm', 'wh', 'reg', 'total'))
    pbar = tqdm(enumerate(train_gen))#, total=train_len, position=0, leave=True)
    for n_batch, d  in pbar:
        step = epoch*n_batch
        images, labels = d 
        
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            loss_dict, total_loss = loss(labels, preds)
            grads = tape.gradient(total_loss, model.trainable_variables)
            if step < hparams['optimizer']['warmup_steps']:
                warmup_opt.apply_gradients(zip(grads, model.trainable_variables))
            else:
                opt.apply_gradients(zip(grads, model.trainable_variables))

        epoch_total_avg(total_loss)
        epoch_hm_avg(loss_dict['hm_loss'])
        epoch_wh_avg(loss_dict['wh_loss'])
        epoch_reg_avg(loss_dict['reg_loss'])
        
#         pbar.set_description('%8.4f\t%8.4f\t%8.4f\t%8.4f'%(loss_dict['hm_loss'], loss_dict['wh_loss'], loss_dict['reg_loss'], total_loss))
        pbar.set_description('%8.4f\t%8.4f\t%8.4f\t%8.4f'%(epoch_hm_avg.result(), epoch_wh_avg.result(), epoch_reg_avg.result(), epoch_total_avg.result()))
        ##Log each 20 step
        if n_batch % 50 == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('total_loss', epoch_total_avg.result(), step=step)
                tf.summary.scalar('hm_loss', epoch_hm_avg.result(), step=step)
                tf.summary.scalar('wh_loss', epoch_wh_avg.result(), step=step)
                tf.summary.scalar('reg_loss', epoch_reg_avg.result(), step=step)
        
        ###
    # END OF EPOCH
    train_gen.on_epoch_end()
    val_gen.on_epoch_end()
    
    train_info['total_loss'][epoch] = epoch_total_avg.result()
    train_info['hm_loss'][epoch] = epoch_hm_avg.result()
    train_info['wh_loss'][epoch] = epoch_wh_avg.result()
    train_info['reg_loss'][epoch] = epoch_reg_avg.result()
    
    #-------------------
    # Evaluate mAP on validation set
    #-------------------
    anno_json = os.path.join(save_model_dir, 'val_label_annotations.json')
    pred_json = os.path.join(save_model_dir, 'val_pred_annotations.json')

#     if not os.path.exists(anno_json):
    generate_coco_format_labels(val_gen, anno_json)
    generate_coco_format_predict(val_gen, model, pred_json, score_threshold=0.01)
    stats = evaluate_all(anno_json, pred_json)
    stats = np.array(stats)
    print('val_mAP@50    | ',end='')
    for i in range(val_gen.num_classes):
        print('Class %d: %8.3f | '%(i, stats[i][1]),end='')
    print('\n')
    print('val_mAP@50:95 | ',end='')
    for i in range(val_gen.num_classes):
        print('Class %d: %8.3f | '%(i, stats[i][0]),end='')
    print('\n')
    
    with val_summary_writer.as_default():
        for i in range(val_gen.num_classes):
            tf.summary.scalar('val_mAP@50_class_%d'%i, stats[i][1], step=epoch)
            tf.summary.scalar('val_mAP@50:95_class_%d'%i, stats[i][0], step=epoch)
        val_mAP50_all = np.mean(stats[:, 1])
        tf.summary.scalar('val_mAP@50_all', val_mAP50_all, step=epoch)
    if val_mAP50_all > best_mAP:
        model_dir = os.path.join(save_model_dir, 'weights_%d_%.4f.h5'%(epoch, val_mAP50_all))
        model.save_weights(model_dir)
        best_mAP = val_mAP50_all
        
    # Save weights each 10 epochs
    if epoch % 10 == 0:
        model_dir = os.path.join(save_model_dir, 'weights_%d_%.4f.h5'%(epoch, val_mAP50_all))
        model.save_weights(model_dir)    
        
#     ## Validation phase
#     val_total_avg = tf.keras.metrics.Mean() # Keeping track of the training loss
#     val_hm_avg = tf.keras.metrics.Mean() # Keeping track of the training accuracy
#     val_wh_avg = tf.keras.metrics.Mean()
#     val_reg_avg = tf.keras.metrics.Mean()
    
#     for n_batch, d in tqdm(enumerate(val_gen), desc='Evaluate validation loss'):
#         images, labels = d 
#         val_preds = model(images, training=False)
#         loss_dict, total_loss = loss(labels, val_preds)  
#         val_total_avg(total_loss)
#         val_hm_avg(loss_dict['hm_loss'])
#         val_wh_avg(loss_dict['wh_loss'])
#         val_reg_avg(loss_dict['reg_loss'])
        
#     val_info['total_loss'][epoch] = val_total_avg.result()
#     val_info['hm_loss'][epoch] = val_hm_avg.result()
#     val_info['wh_loss'][epoch] = val_wh_avg.result()
#     val_info['reg_loss'][epoch] = val_reg_avg.result()

#     print(f"Epoch: [{epoch}]\t", {k:'%3f'%val_info[k][epoch] for k in val_info.keys()} )
#     ## Author said that even though validation loss become higher, AP still better
#     ## So save weights each epoch instead of best on validation
# #     if current_val_loss > val_info['total_loss'][epoch]:
#     model_dir = os.path.join(save_model_dir, 'weights_%d_%.4f.h5'%(epoch, val_info['total_loss'][epoch]))
#     model.save_weights(model_dir)
#     current_val_loss = val_info['total_loss'][epoch]
        
#     with val_summary_writer.as_default():
#         tf.summary.scalar('val_total_loss', val_total_avg.result(), step=epoch)
#         tf.summary.scalar('val_hm_loss', val_hm_avg.result(), step=epoch)
#         tf.summary.scalar('val_wh_loss', val_wh_avg.result(), step=epoch)
#         tf.summary.scalar('val_reg_loss', val_reg_avg.result(), step=epoch)
        
# print(train_info)
# print(val_info)

             
    
