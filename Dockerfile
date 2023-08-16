FROM python:3.10
ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN python -m pip install --upgrade pip
# -----------------------------------
# create required folder
RUN mkdir /app
# RUN mkdir /app/speaker_verification
# -----------------------------------
# Copy required files from repo into image
COPY ./pretrained_models /app/pretrained_models
COPY ./samples /app/samples
COPY ./audio.py /app/
COPY ./batcher.py /app/
COPY ./cli.py /app/
COPY ./cli_updated.py /app/
COPY ./constants.py /app/
COPY ./conv_models.py /app/
COPY ./deep-speaker /app/
COPY ./ResCNN_softmax_pre_training_checkpoint_102.h5 /app/
COPY ./ResCNN_triplet_training_checkpoint_265.h5 /app/
COPY ./testing.py /app/
COPY ./triplet_loss.py /app/
COPY ./utils.py /app/
COPY ./app.py /app/
COPY ./eval_metrics.py /app/
COPY ./pyproject.toml /app/
# -----------------------------------
# switch to application directory
WORKDIR /app
# -----------------------------------
# update image os
RUN python -m pip install --upgrade pip
RUN apt-get update
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install
# -----------------------------------
# environment variables
ENV PYTHONUNBUFFERED=1
# -----------------------------------
# run the app (re-configure port if necessary)
#ENTRYPOINT ["python", "Speaker_Verification.py"]
#CMD ["python", "Speaker_Verification.py"]

EXPOSE 7070
ENTRYPOINT ["python"]
CMD ["app.py"]