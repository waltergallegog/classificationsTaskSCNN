import tensorflow as tfLocal


class Masking(tfLocal.keras.constraints.Constraint):
  def __init__(self, mask=0):
    self.mask = mask

  def __call__(self, w):
    return tfLocal.multiply(w, self.mask)


def Relu(inputs):
    S = 201.0
    tau_syn = 0.005
    return tfLocal.multiply(S*tau_syn, tfLocal.math.maximum(0.0, inputs))


def netModelsComplete(structure, dataShape, datasetClass):
    modelCNN = None
    if structure == 'c06c12f2':
        modelCNN = tfLocal.keras.models.Sequential([
            tfLocal.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=Relu, use_bias=False, input_shape=dataShape),
            tfLocal.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tfLocal.keras.layers.Conv2D(filters=12, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=Relu, use_bias=False),
            tfLocal.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tfLocal.keras.layers.Flatten(),
            tfLocal.keras.layers.Dense(units=768, activation='relu', use_bias=False),
            tfLocal.keras.layers.Dense(units=datasetClass, activation='softmax', use_bias=False)
        ])
    elif structure == 'c12c24f2':
        modelCNN = tfLocal.keras.models.Sequential([
            tfLocal.keras.layers.Conv2D(filters=12, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=Relu, use_bias=False, input_shape=dataShape),
            tfLocal.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tfLocal.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=Relu, use_bias=False),
            tfLocal.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tfLocal.keras.layers.Flatten(),
            tfLocal.keras.layers.Dense(units=768, activation='relu', use_bias=False),
            tfLocal.keras.layers.Dense(units=datasetClass, activation='softmax', use_bias=False)
        ])
    else:
        raise Exception('Inserted structure not available')

    return modelCNN


def netModelsPruned(structure, dataShape, mask, datasetClass):
    modelCNN = None
    if structure == 'c06c12f2':
        modelCNN = tfLocal.keras.models.Sequential([
            tfLocal.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=Relu, use_bias=False, kernel_constraint=Masking(mask[0]), input_shape=dataShape),
            tfLocal.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tfLocal.keras.layers.Conv2D(filters=12, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=Relu, use_bias=False, kernel_constraint=Masking(mask[1])),
            tfLocal.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tfLocal.keras.layers.Flatten(),
            tfLocal.keras.layers.Dense(units=768, activation='relu', use_bias=False, kernel_constraint=Masking(mask[2])),
            tfLocal.keras.layers.Dense(units=datasetClass, activation='softmax', use_bias=False, kernel_constraint=Masking(mask[3]))
        ])
    elif structure == 'c12c24f2':
        modelCNN = tfLocal.keras.models.Sequential([
            tfLocal.keras.layers.Conv2D(filters=12, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=Relu, use_bias=False, kernel_constraint=Masking(mask[0]), input_shape=dataShape),
            tfLocal.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tfLocal.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=Relu, use_bias=False, kernel_constraint=Masking(mask[1])),
            tfLocal.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tfLocal.keras.layers.Flatten(),
            tfLocal.keras.layers.Dense(units=768, activation='relu', use_bias=False, kernel_constraint=Masking(mask[2])),
            tfLocal.keras.layers.Dense(units=datasetClass, activation='softmax', use_bias=False, kernel_constraint=Masking(mask[3]))
        ])
    else:
        raise Exception('Inserted structure not available')

    return modelCNN