import tensorflow as tf

def huber_loss(labels, predictions, delta=1.0):
    # Absolute value
    residual = tf.abs(predictions - labels)
    
    # Conditional checking
    condition = tf.less(residual, delta)
    
    # More operations, we need to both of the conditional
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    
    # Check the condition and output the exact result
    result = tf.where(condition, small_res, large_res)
    return result
