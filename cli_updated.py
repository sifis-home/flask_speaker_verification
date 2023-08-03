#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


from constants import SAMPLE_RATE
from testing import test

logger = logging.getLogger(__name__)

VERSION = "3.0a"
working_dir = "C:/Users/wisam/Desktop/deep-speaker-master/.deep-speaker-wd/triplet-training/"
# audio_dir = 'C:/Users/wisam/Desktop/deep-speaker-master/LibriSpeech/test-clean/61/70968'
sample_rate = SAMPLE_RATE
counts_per_speaker = 600, 100
checkpoint_file = "ResCNN_triplet_training_checkpoint_265.h5"
pre_training_phase = False


def test_model(working_dir, checkpoint_file):
    # export CUDA_VISIBLE_DEVICES=0; python cli.py test-model
    # --working_dir /home/philippe/ds-test/triplet-training/
    # --checkpoint_file ../ds-test/checkpoints-softmax/ResCNN_checkpoint_102.h5
    # f-measure = 0.789, true positive rate = 0.733, accuracy = 0.996, equal error rate = 0.043

    # export CUDA_VISIBLE_DEVICES=0; python cli.py test-model
    # --working_dir /home/philippe/ds-test/triplet-training/
    # --checkpoint_file ../ds-test/checkpoints-triplets/ResCNN_checkpoint_175.h5
    # f-measure = 0.849, true positive rate = 0.798, accuracy = 0.997, equal error rate = 0.025
    test(working_dir, checkpoint_file)


if __name__ == "__main__":
    test_model(working_dir, checkpoint_file)
    # cli()
