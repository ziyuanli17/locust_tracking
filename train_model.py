from deepposekit.io import TrainingGenerator, DataGenerator
from deepposekit.augment import FlipAxis
from deepposekit.callbacks import Logger, ModelCheckpoint
from deepposekit.models import DeepLabCut, StackedDenseNet, StackedHourglass, LEAP, load_model
from tensorflow_core.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping

import imgaug.augmenters as iaa
import imgaug as ia
import numpy as np
import time


# Define callbacks to enhance model training
def get_callbacks():
    logger = Logger(validation_batch_size=10,
                    # filepath saves the logger data to a .h5 file
                    # filepath=HOME + "/deepposekit-data/datasets/fly/log_densenet.h5"
                    )

    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, verbose=1, patience=20)

    model_checkpoint = ModelCheckpoint(
       "best_model_densenet.h5",
        monitor="loss",
        verbose=1,
        save_best_only=True,
    )

    early_stop = EarlyStopping(
        monitor="loss",
        # monitor="loss" # use if validation_split=0
        min_delta=0.001,
        patience=100,
        verbose=1
    )

    return [early_stop, reduce_lr, model_checkpoint, logger]


# The main function to train model
def train_model(data_generator, resume=False):
    # Create an augmentation pipeline
    augmenter = []

    augmenter.append(FlipAxis(data_generator, axis=0))  # flip image up-down
    augmenter.append(FlipAxis(data_generator, axis=1))  # flip image left-right

    sometimes = [iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                            shear=(-8, 8),
                            order=ia.ALL,
                            cval=ia.ALL,
                            mode=ia.ALL), iaa.Affine(scale=(0.8, 1.2),
                                                     mode=ia.ALL,
                                                     order=ia.ALL,
                                                     cval=ia.ALL)]
    augmenter.append(iaa.Sometimes(0.75, sometimes))
    augmenter.append(iaa.Affine(rotate=(-180, 180),
                                mode=ia.ALL,
                                order=ia.ALL,
                                cval=ia.ALL)
                     )
    augmenter = iaa.Sequential(augmenter)

    # Define ore resume a model
    if resume:
        model = load_model(
            "best_model_densenet.h5",
            augmenter=augmenter,
            generator=data_generator,
        )
    else:
        # Create a TrainingGenerator
        train_generator = TrainingGenerator(generator=data_generator,
                                            downsample_factor=3,
                                            augmenter=augmenter,
                                            sigma=5,
                                            validation_split=0,
                                            use_graph=True,
                                            random_seed=1,
                                            graph_scale=1)

        model = StackedDenseNet(train_generator, n_stacks=2, growth_rate=32, pretrained=True)

    callbacks = get_callbacks()

    # Fit the model
    model.fit(
        batch_size=16,
        validation_batch_size=10,
        callbacks=callbacks,
        # epochs=1000, # Increase the number of epochs to train the model longer
        epochs=200,
        n_workers=8,
        steps_per_epoch=None,
    )


if __name__ == '__main__':
    data_generator = DataGenerator('annotation_set.h5')
    train_model(data_generator)
