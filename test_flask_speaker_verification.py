from unittest.mock import patch

import numpy as np
import requests

from app import load_model, on_close, on_error, on_open
from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import NUM_FRAMES, SAMPLE_RATE
from conv_models import DeepSpeakerModel
from testing import batch_cosine_similarity

# def test_create_sequences():
#     values = [1, 2, 3, 4, 5]
#     time_steps = 3
#     result = create_sequences(values, time_steps)
#     expected_result = [
#         [1, 2, 3],
#         [2, 3, 4],
#         [3, 4, 5],
#     ]
#     assert result.tolist() == expected_result


# def test_batch_cosine_similarity():
#     x1 = 35
#     x2 = 45
#     similarity = batch_cosine_similarity(x1, x2)

#     mul = np.multiply(x1, x2)
#     expected_similarity = np.sum(mul, axis=1)

#     assert similarity == expected_similarity


def test_():
    first_file = "samples/PhilippeRemy_001.wav"
    second_file = "samples/PhilippeRemy_002.wav"
    third_file = "samples/sample1.wav"

    model = load_model()

    mfcc_001 = sample_from_mfcc(read_mfcc(first_file, SAMPLE_RATE), NUM_FRAMES)
    mfcc_002 = sample_from_mfcc(
        read_mfcc(second_file, SAMPLE_RATE), NUM_FRAMES
    )
    mfcc_003 = sample_from_mfcc(read_mfcc(third_file, SAMPLE_RATE), NUM_FRAMES)

    # Call the model to get the embeddings of shape (1, 512) for each file.
    predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
    predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))
    predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

    output1 = batch_cosine_similarity(predict_001, predict_002)
    output2 = batch_cosine_similarity(predict_001, predict_003)

    assert type(output1) == type(output2)


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
