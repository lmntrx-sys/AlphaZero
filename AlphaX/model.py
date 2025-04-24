import tensorflow as tf
from tensorflow.keras import layers # type: ignore

def residual_block(x, filters, strides=1):
    res = x
    x = layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if strides > 1 or x.shape[-1] != res.shape[-1]:
        res = layers.Conv2D(filters, 1, strides=strides, padding='same')(res)
        res = layers.BatchNormalization()(res)

    x = layers.Add()([x, res])
    x = layers.ReLU()(x)
    return x

def policy_head(x, num_actions):
    x = layers.Flatten()(x)
    x = layers.Dense(num_actions, activation='softmax')(x)
    return x

def value_head(x):
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(1, activation='tanh')(x)
    return x

def create_resnet_tictactoe_model(input_shape, num_actions):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(32, 3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    policy = policy_head(x, num_actions)
    value = value_head(x)

    model = tf.keras.Model(inputs=inputs, outputs=[policy, value])

    return model

policy_model = create_resnet_tictactoe_model((1, 3, 3, 3), 9)()
policy_model.summary()