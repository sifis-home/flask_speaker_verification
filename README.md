# WP4 Analytic: Privacy-Aware Speaker Verification
Speaker verification is a process that involves authenticating individuals based on the unique biometric aspect of their voice. This approach offers a non-intrusive and secure method for identity verification. Within the SIFIS-home environment, accurately identifying individuals is crucial for granting appropriate privileges based on predefined policies. To achieve this, a speaker verification system analyzes the voices of shared audio files and communicates the identified individuals to the smart-home components to grant or revoke access. 
The Privacy-Aware Speaker Recognition and Verification system requires two audio files containing voice as input data for verification. The analytic is designed to process WAV or FLAC audio samples with specific requirements, including a sampling rate of 16 kHz or 8 kHz, a single channel (mono) audio, and the duration of the audio segment should be within a certain range, typically a few seconds. If the audio sample is in a different format, a preprocessing step may be necessary to adjust it to meet the input requirements of the analytic. 
For speaker verification, we use ECAPA-TDN model, developed as part of the [SpeechBrain AI toolkit](https://github.com/speechbrain/speechbrain/tree/develop). ECAPA-TDNN model employs ECAPA Time Delay Neural Networks (TDNNs) derived embeddings, and it consists of an input layer, followed by a convolutional block with ReLU activation and batch normalization. Then, a sequence of three Squeeze-and-Excitation and residual blocks. Next, a convolutional block with ReLU activation. Followed by a layer that applies statistics pooling to project variable-length utterances into fixed-length speaker characterizing embeddings with batch normalization. Then a fully connected dense layer with batch normalization, and an Additive Angular Margin (AAM) Softmax layer. Finally, an output layer to classify the inputs as yes or no for verification results. 
The output of this analytic is a binary decision indicating whether the two input audio samples belong to the same speaker or not. It evaluates the similarity or dissimilarity between the input sample and the enrolled speaker's reference data in the other sample. The output is represented as a similarity metric using cosine similarity

Since applying privacy mechanisms would alter the speaker’s voice in the audio files, the protection mechanisms that can be applied with this analytic include file encryption and employing secure protocols for transmitting voice data. 

## Deploying

### Privacy-Aware Speaker Verification in a container

Privacy-Aware Speaker Verification is intended to run in a docker container on port 7070. The Dockerfile at the root of this repo describes the container. To build and run it execute the following commands:

`docker build -t flask_speaker_verification .`

`docker-compose up`

## REST API of Privacy-Aware Speaker Verification

Description of the REST endpoint available while Privacy-Aware Speaker Verification is running.

---

#### GET /speaker_verification

Description: The output of this analytic is a binary decision indicating whether the two input audio samples belong to the same speaker or not.

Command: 

`curl -X POST -F "file1=@file1_location.wav;type=audio/wav" -F "file2=@file2_location.wav;type=audio/wav" http://localhost:7070/speaker_verification/<first_audio_file.wav>/<second_audio_file.wav>/<epsilon>/<sensitivity>/<requestor_id>/<requestor_type>/<request_id>`

Sample: 

`curl -X POST -F  "file1=@file1_location.wav;type=audio/wav" -F "file2=@file2_location.wav;type=audio/wav"  http://localhost:7070/speaker_verification/first_audio_file.wav/second_audio_file.wav/33466553786f48cb72faad7b2fb9d0952c97/NSSD/2023061906001633466553786f48cb72faad7b2fb9d0952c97`


---
## License

Released under the [MIT License](LICENSE).

## Acknowledgements

This software has been developed in the scope of the H2020 project SIFIS-Home with GA n. 952652.
