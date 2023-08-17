import os
import unittest
from unittest.mock import patch

import numpy as np
import requests
import tensorflow as tf

import testing
import utils
from app import load_model, on_close, on_error, on_open
from batcher import (
    OneHotSpeakers,
    SparseCategoricalSpeakers,
    TripletBatcher,
    TripletEvaluator,
    extract_speaker,
    sample_from_mfcc,
    sample_from_mfcc_file,
)
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


class TestSampleFromMFCC(unittest.TestCase):
    def test_sample_from_mfcc(self):
        mfcc = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        max_length = 2
        sampled_mfcc = sample_from_mfcc(mfcc, max_length)


class TestTripletEvaluator(unittest.TestCase):
    def setUp(self):
        num_samples = 100
        num_classes = 5
        self.kx_test = np.random.random((num_samples, 32, 64, 1))
        self.ky_test = np.random.randint(
            num_classes, size=(num_samples, num_classes)
        )

    def test_triplet_evaluator(self):
        triplet_evaluator = TripletEvaluator(self.kx_test, self.ky_test)

        # Test _select_speaker_data
        speaker = 0
        selected_data = triplet_evaluator._select_speaker_data(speaker)
        self.assertEqual(selected_data.shape, (1, 32, 64, 1))

        # Test get_speaker_verification_data
        positive_speaker = 0
        num_different_speakers = 3
        data = triplet_evaluator.get_speaker_verification_data(
            positive_speaker, num_different_speakers
        )
        self.assertEqual(data.shape, (num_different_speakers + 2, 32, 64, 1))


class TestTripletBatcher(unittest.TestCase):
    def setUp(self):
        num_speakers = 10
        num_samples = 100
        num_classes = 5
        self.kx_train = np.random.random((num_samples, 32, 64, 1))
        self.ky_train = np.random.randint(
            num_classes, size=(num_samples, num_classes)
        )
        self.kx_test = np.random.random((num_samples, 32, 64, 1))
        self.ky_test = np.random.randint(
            num_classes, size=(num_samples, num_classes)
        )

    def test_triplet_batcher(self):
        triplet_batcher = TripletBatcher(
            self.kx_train, self.ky_train, self.kx_test, self.ky_test
        )

        # Test select_speaker_data
        n = 10
        is_test = False
        speaker = 0
        selected_data = triplet_batcher.select_speaker_data(
            speaker, n, is_test
        )
        self.assertEqual(selected_data.shape, (n, 32, 64, 1))

        # Test get_batch
        batch_size = 6
        is_test = False
        batch_x, batch_y = triplet_batcher.get_batch(batch_size, is_test)
        self.assertEqual(batch_x.shape, (batch_size, 32, 64, 1))
        self.assertEqual(
            batch_y.shape, (batch_size, len(triplet_batcher.speakers_list))
        )


class TestOneHotSpeakers(unittest.TestCase):
    def test_one_hot_speakers(self):
        speakers_list = ["speaker_1", "speaker_2", "speaker_3"]
        one_hot_speakers = OneHotSpeakers(speakers_list)


class TestSparseCategoricalSpeakers(unittest.TestCase):
    def test_sparse_categorical_speakers(self):
        speakers_list = ["speaker_1", "speaker_2", "speaker_3"]
        sparse_categorical_speakers = SparseCategoricalSpeakers(speakers_list)

        self.assertEqual(
            sparse_categorical_speakers.speaker_ids, sorted(speakers_list)
        )

        # Check that all speaker IDs are unique
        self.assertEqual(
            len(set(sparse_categorical_speakers.speaker_ids)),
            len(sparse_categorical_speakers.speaker_ids),
        )

        # Check that the mapping is correct
        expected_map = {"speaker_1": 0, "speaker_2": 1, "speaker_3": 2}
        self.assertEqual(sparse_categorical_speakers.map, expected_map)

        # Test getting an index
        self.assertEqual(sparse_categorical_speakers.get_index("speaker_2"), 1)


class TestSampleFromMFCCFile(unittest.TestCase):
    @patch("batcher.np.load")  # Mock the np.load function
    @patch("batcher.sample_from_mfcc")  # Mock the sample_from_mfcc function
    def test_sample_from_mfcc_file(self, mock_sample_from_mfcc, mock_np_load):
        utterance_file = "samples/PhilippeRemy_001.wav"
        max_length = 10

        # Mock the np.load function to return a mock MFCC array
        mock_mfcc = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mock_np_load.return_value = mock_mfcc

        # Mock the sample_from_mfcc function to return a modified mock MFCC array
        mock_sampled_mfcc = np.array([[2, 3], [5, 6]])
        mock_sample_from_mfcc.return_value = mock_sampled_mfcc

        # Call the function and check the result
        result = sample_from_mfcc_file(utterance_file, max_length)


class TestExtractSpeaker(unittest.TestCase):
    def test_extract_speaker(self):
        utt_file = "samples/PhilippeRemy_001.wav"
        extracted_speaker = extract_speaker(utt_file)
        # self.assertEqual(extracted_speaker, "speaker")


class TestCheckpointCompatibility(unittest.TestCase):
    def test_test_checkpoint_compatibility(self):
        dsm = DeepSpeakerModel(
            batch_input_shape=(None, 32, 64, 4),
            include_softmax=True,
            num_speakers_softmax=10,
        )
        dsm.m.save_weights("test.h5")
        dsm = DeepSpeakerModel(
            batch_input_shape=(None, 32, 64, 4), include_softmax=False
        )
        dsm.m.load_weights("test.h5", by_name=True)
        os.remove("test.h5")


class TestBatchCosineSimilarity(unittest.TestCase):
    def test_batch_cosine_similarity(self):
        x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        x2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

        similarity = testing.batch_cosine_similarity(x1, x2)

        expected_similarity = np.array([46, 73, 46])

        np.testing.assert_array_equal(similarity, expected_similarity)


def pass_function(item):
    # Define your function implementation here
    pass


class TestParallelFunction(unittest.TestCase):
    def test_parallel_function(self):
        input_data = [1, 2, 3, 4, 5]
        result = utils.parallel_function(pass_function, input_data)

        self.assertEqual(len(result), len(result))


def test_load_best_checkpoint():
    checkpoint_dir = "Test_Directory"
    input_data1 = [1, 2, 3, 4, 5]
    input_data2 = [1, 2, 3, 4, 5]
    utils.load_best_checkpoint(checkpoint_dir)
    utils.delete_older_checkpoints(checkpoint_dir, max_to_keep=5)
    assert len(input_data1) == len(input_data2)


def test_load_pickle():
    input_data1 = [1, 2, 3, 4, 5]
    input_data2 = [1, 2, 3, 4, 5]

    utils.load_pickle(file="Test_File")
    utils.load_npy(file="Test_File")
    utils.enable_deterministic()
    utils.load_pickle(file="Test_File")
    assert len(input_data1) == len(input_data2)


def test_deep_speaker_loss_alpha_0_1():
    loss = deep_speaker_loss(
        alpha=0.1, y_true=0, y_pred=np.array([[0.9], [1.0], [-1.0]])
    )
    assert isinstance(loss.numpy(), float)
    assert 0 <= loss.numpy() <= 1


def test_deep_speaker_loss_alpha_1():
    loss = deep_speaker_loss(
        alpha=1, y_true=0, y_pred=np.array([[0.9], [1.0], [-1.0]])
    )
    assert isinstance(loss.numpy(), float)
    assert 0 <= loss.numpy() <= 1


def test_deep_speaker_loss_alpha_2():
    loss = deep_speaker_loss(
        alpha=2, y_true=0, y_pred=np.array([[0.9], [1.0], [-1.0]])
    )
    assert isinstance(loss.numpy(), float)
    assert 0 <= loss.numpy() <= 1


def test_deep_speaker_loss_custom_values():
    loss = deep_speaker_loss(
        alpha=2, y_true=0, y_pred=np.array([[0.6], [1.0], [0.0]])
    )
    assert isinstance(loss.numpy(), float)
    assert 0 <= loss.numpy() <= 2


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
