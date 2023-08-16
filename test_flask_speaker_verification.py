from unittest.mock import patch

import numpy as np
import requests
import tensorflow as tf

import utils
from app import load_model, on_close, on_error, on_open
from constants import (
    BATCH_SIZE,
    CHECKPOINTS_SOFTMAX_DIR,
    CHECKPOINTS_TRIPLET_DIR,
    NUM_FBANKS,
    NUM_FRAMES,
    SAMPLE_RATE,
    TRAIN_TEST_RATIO,
)
from conv_models import DeepSpeakerModel
from eval_metrics import (
    calculate_accuracy,
    calculate_eer,
    calculate_roc,
    calculate_val_far,
    evaluate,
)
from triplet_loss import batch_cosine_similarity, deep_speaker_loss

# Test data for sims and labels
sims = np.array([0.8, 0.6, 0.7, 0.9, 0.5])
labels = np.array([1, 0, 1, 1, 0])
batch_size = 6
embedding_size = 512
y_true = np.random.rand(batch_size, embedding_size)
y_pred = np.random.rand(batch_size, embedding_size)


def test_batch_cosine_similarity():
    tf_y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    cosine_sim = batch_cosine_similarity(tf_y_pred, tf_y_pred)
    assert isinstance(cosine_sim, tf.Tensor)
    assert cosine_sim.shape == (batch_size,)


def test_deep_speaker_loss():
    tf_y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    tf_y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    loss = deep_speaker_loss(tf_y_true, tf_y_pred)
    assert isinstance(loss, tf.Tensor)


def test_evaluate():
    fm, tpr, acc, eer = evaluate(sims, labels)
    assert isinstance(fm, float)
    assert isinstance(tpr, float)
    assert isinstance(acc, float)
    assert isinstance(eer, float)


def test_calculate_roc():
    thresholds = np.arange(0, 1.0, 0.001)
    bestfm, besttpr, bestacc = calculate_roc(thresholds, sims, labels)
    assert isinstance(bestfm, float)
    assert isinstance(besttpr, float)
    assert isinstance(bestacc, float)


threshold = 0.6
sims = np.array([0.8, 0.6, 0.7, 0.9, 0.5])
labels = np.array([1, 0, 1, 1, 0])
actual_issame = np.array([1, 0, 1, 1, 0])


def test_calculate_accuracy():
    tpr, fpr, precision, fm, acc = calculate_accuracy(threshold, sims, labels)
    assert isinstance(tpr, float)
    assert isinstance(fpr, float)
    assert isinstance(precision, float)
    assert isinstance(fm, float)
    assert isinstance(acc, float)


def test_calculate_eer():
    thresholds = np.arange(0, 1.0, 0.001)
    eer = calculate_eer(thresholds, sims, labels)
    assert isinstance(eer, float)


def test_calculate_val_far():
    frr, far = calculate_val_far(threshold, sims, actual_issame)
    assert isinstance(frr, float)
    assert isinstance(far, float)


def test_utils():
    utils.find_files(directory="samples/", ext="wav")
    utils.init_pandas()
    utils.create_new_empty_dir(directory="Test_Directory")
    utils.ensure_dir_for_filename(filename="Test_File")
    utils.ensures_dir(directory=CHECKPOINTS_SOFTMAX_DIR)
    utils.ensures_dir(directory=CHECKPOINTS_TRIPLET_DIR)
    x = TRAIN_TEST_RATIO + SAMPLE_RATE
    y = TRAIN_TEST_RATIO * NUM_FBANKS * BATCH_SIZE
    z = NUM_FRAMES
    assert z == NUM_FRAMES


# def test_batch_cosine_similarity():
#     x1 = 35
#     x2 = 45
#     similarity = batch_cosine_similarity(x1, x2)

#     mul = np.multiply(x1, x2)
#     expected_similarity = np.sum(mul, axis=1)

#     assert similarity == expected_similarity


# def test_verification():
#     first_file = "samples/PhilippeRemy_001.wav"
#     second_file = "samples/PhilippeRemy_002.wav"
#     third_file = "samples/sample1.wav"

#     model = load_model()

#     mfcc_001 = sample_from_mfcc(read_mfcc(first_file, SAMPLE_RATE), NUM_FRAMES)
#     mfcc_002 = sample_from_mfcc(
#         read_mfcc(second_file, SAMPLE_RATE), NUM_FRAMES
#     )
#     mfcc_003 = sample_from_mfcc(read_mfcc(third_file, SAMPLE_RATE), NUM_FRAMES)

#     # # Call the model to get the embeddings of shape (1, 512) for each file.
#     # predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
#     # predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))
#     # predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

#     # output1 = batch_cosine_similarity(predict_001, predict_002)
#     # output2 = batch_cosine_similarity(predict_001, predict_003)

#     assert type(mfcc_001) == type(mfcc_002)


def test_load_model():
    model = load_model()

    # Define the model here.
    expected_model = DeepSpeakerModel()

    # URL of the pretrained model weights
    weights_url = (
        "https://drive.google.com/uc?id=1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP"
    )

    # Local path to save the downloaded weights file
    local_weights_path = "ResCNN_triplet_training_checkpoint_265.h5"

    # Download the weights using requests
    response = requests.get(weights_url)
    if response.status_code == 200:
        with open(local_weights_path, "wb") as f:
            f.write(response.content)
        print("Weights downloaded and saved successfully.")
    else:
        print("Failed to download weights.")

    expected_model.m.load_weights(
        "ResCNN_triplet_training_checkpoint_265.h5", by_name=True
    )

    assert type(model) == type(expected_model)


def test_on_error():
    error = "WebSocket error occurred"

    with patch("builtins.print") as mock_print:
        on_error(None, error)

    mock_print.assert_called_once_with(error)


def test_on_close():
    close_status_code = 1000
    close_msg = "Connection closed"

    with patch("builtins.print") as mock_print:
        on_close(None, close_status_code, close_msg)

    mock_print.assert_called_once_with("### Connection closed ###")


def test_on_open():
    with patch("builtins.print") as mock_print:
        on_open(None)

    mock_print.assert_called_once_with("### Connection established ###")
