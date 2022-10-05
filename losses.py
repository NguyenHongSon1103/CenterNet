import tensorflow as tf

LOSS_KEYS = ['hm_loss', 'wh_loss', 'reg_loss', 'total_loss']

def penalty_sigmoid_focal(hm_pred, hm_true, alpha=2.0, beta=4.0):
    is_present_tensor = tf.math.equal(hm_true, 1.0)
    prediction_tensor = tf.clip_by_value(hm_pred, 1e-4,
                                         1 - 1e-4)

    positive_loss = (tf.math.pow((1 - prediction_tensor), alpha)*
                                     tf.math.log(prediction_tensor))
    negative_loss = (tf.math.pow((1 - hm_true), beta)*
                                     tf.math.pow(prediction_tensor, alpha)*
                                     tf.math.log(1 - prediction_tensor))

    loss = -tf.where(is_present_tensor, positive_loss, negative_loss)
    num_pos = tf.reduce_sum(tf.cast(is_present_tensor, tf.float32))
    return tf.reduce_sum(loss)/num_pos

def mse_loss(ypred, ytrue):
    mse = tf.keras.losses.mean_squared_error(ytrue, ypred)
    return tf.reduce_mean(mse)
    
def balanced_crossentropy_loss(pred, gt, negative_ratio=3.):
    positive_mask, negative_mask = gt[..., 0], 1-gt[..., 0]
    positive_count = tf.reduce_sum(positive_mask)
    negative_count = tf.reduce_min([tf.reduce_sum(negative_mask), positive_count * negative_ratio])
    loss = tf.keras.losses.binary_crossentropy(gt, pred)
    positive_loss = loss * positive_mask
    negative_loss = loss * negative_mask
    negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, (-1,)), tf.cast(negative_count, tf.int32))

    balanced_loss = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (
            positive_count + negative_count + 1e-6)
    balanced_loss = balanced_loss
    return balanced_loss

def l1_loss(ytrue, ypred):
    mask = tf.cast(ytrue > 0.0, tf.float32)
    num_pos = tf.reduce_sum(mask)
    total_loss = tf.reduce_sum(tf.abs(ytrue*mask - ypred*mask))
    return total_loss / (num_pos + 1e-5)

def loss(ytrue, ypred):
    wh_pred, reg_pred, hm_pred = ypred[..., :2], ypred[..., 2:4], ypred[..., 4:]
    wh_true, reg_true, hm_true = ytrue[..., :2], ytrue[..., 2:4], ytrue[..., 4:]
    hm_loss = penalty_sigmoid_focal(hm_pred, hm_true)
    wh_loss = 0.1 * l1_loss(wh_true, wh_pred)
    reg_loss = l1_loss(reg_true, reg_pred)
    total_loss = hm_loss + wh_loss + reg_loss
    loss_dict = {key:value for key, value in zip(LOSS_KEYS, [hm_loss, wh_loss, reg_loss, total_loss])}
    
    return loss_dict, total_loss


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    size = 64
    nums_cls = 1
    hm_pred = np.random.random((5, 64, 64, nums_cls)).astype('float32')
    wh_pred = np.random.random((5, 64, 64, 2)).astype('float32')
    reg_pred = np.random.random((5, 64, 64, 2)).astype('float32')
    hm_true = np.random.random((5, 64, 64, nums_cls)).astype('float32')
    wh_true = np.random.random((5, 100, 2)).astype('float32')
    reg_true = np.random.random((5, 100, 2)).astype('float32')
    reg_mask = np.random.random((5, 100)).astype('float32')
    indices = np.random.random((5, 100)).astype('float32')
    
    ypred = [hm_pred, wh_pred, reg_pred]
    ytrue = [hm_true, wh_true, reg_true, reg_mask, indices]
#     print(loss(ypred + ytrue))
    
    ytrue = np.random.random((5, 64, 64, nums_cls+4)).astype('float32')
    ypred = np.random.random((5, 64, 64, nums_cls+4)).astype('float32')
    print(loss(ytrue, ypred))
