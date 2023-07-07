import random
import numpy as np
from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from testing import batch_cosine_similarity
from flask import Flask, abort, request
from tempfile import NamedTemporaryFile
import datetime
import hashlib
import re
import os
import platform

import random
from pydub import AudioSegment
import io

import websocket
import json
import _thread
import rel

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
model = DeepSpeakerModel()

model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

app = Flask(__name__)

def on_error(ws, error):
    print(error)


def on_close(ws, close_status_code, close_msg):
    print("### Connection closed ###")


def on_open(ws):
    print("### Connection established ###")

@app.route('/speaker_verification/<first_audio_file>/<second_audio_file>/<requestor_id>/<requestor_type>/<request_id>', methods=['POST'])
def handler(first_audio_file,second_audio_file,requestor_id,requestor_type,request_id):
    if not request.files:
        # If the user didn't submit any files, return a 400 (Bad Request) error.
        abort(400)

    # Initialize empty lists for storing file names and handles
    file_names = []
    file_handles = []
    output = 0.0
    result = ""

    for key, file in request.files.items():
        print(key)
        # Create a temporary file.
        temp_file = NamedTemporaryFile(delete=False)
        # Save the user's uploaded file to the temporary file.
        file.save(temp_file)
        print(temp_file.name)
        # Add the file name and handle to the lists.
        file_names.append(temp_file.name)
        file_handles.append(temp_file)

    # Process the uploaded files here...
    print(file_names)
    ### Encryption and Decryption for the files ###
    mfcc_001 = sample_from_mfcc(read_mfcc(file_names[0], SAMPLE_RATE), NUM_FRAMES)
    mfcc_002 = sample_from_mfcc(read_mfcc(file_names[1], SAMPLE_RATE), NUM_FRAMES)


    # Call the model to get the embeddings of shape (1, 512) for each file.
    predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
    predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

    output = batch_cosine_similarity(predict_001, predict_002)
    if output > 0.6:
        result = "Same Speaker"
    else:
        result = "Different Speaker"

    print(result)

    # Close the file handles and delete the temporary files.
    for handle in file_handles:
        handle.close()
        os.unlink(handle.name)

    analyzer_id = platform.node()
    print(analyzer_id)

    # Get current date and time
    now = datetime.datetime.now()

    # Generate a random hash using SHA-256 algorithm
    hash_object = hashlib.sha256()
    hash_object.update(bytes(str(now), 'utf-8'))
    hash_value = hash_object.hexdigest()

    # Concatenate the time and the hash
    analysis_id = str(analyzer_id) + str(now) + hash_value

    ws_req_final = {
                    "RequestPostTopicUUID": {
                    "topic_name": "SIFIS:Privacy_Aware_Speaker_Verification_Results",
                    "topic_uuid": "Speaker_Verification_Results",
                    "value": {
                        "description": "Speaker Verification Results",
                        "requestor_id": str(requestor_id),
                        "requestor_type": str(requestor_type),
                        "request_id": str(request_id),
                        "analyzer_id": str(analyzer_id),
                        "analysis_id": str(analysis_id),
                        "connected": True,
                        "first_audio_file": str(first_audio_file),
                        "second_audio_file": str(second_audio_file),
                        "Cosine Similarity": float(output),
                        "Result": str(result)
                    }
                }
            }

    # return result
    ws.send(json.dumps(ws_req_final))
    return ws_req_final

if __name__ == "__main__":
    ws = websocket.WebSocketApp("ws://localhost:3000/ws",
                                on_open=on_open,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever(dispatcher=rel)  # Set dispatcher to automatic reconnection
    rel.signal(2, rel.abort)  # Keyboard Interrupt
    app.run(host='0.0.0.0', port=7070)