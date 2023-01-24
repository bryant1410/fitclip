#!/usr/bin/env python
import sys

from google.cloud.speech_v1p1beta1 import RecognitionAudio, RecognitionConfig, RecognitionMetadata, \
    SpeakerDiarizationConfig, SpeechClient


# We use a Python script as the `gcloud` equivalent command doesn't support the enhanced models
# (`gcloud ml speech recognize-long-running`, not even the alpha and beta ones).
# See https://cloud.google.com/sdk/gcloud/reference/alpha/ml/speech/recognize-long-running for more info.
# noinspection PyTypeChecker
def main() -> None:
    assert len(sys.argv) == 2, f"Valid syntax: {sys.argv[0]} GS_PATH"

    path = sys.argv[1]  # -Cv9h3ic2JI.opus, czQwCto9O80.opus, mono: --BLA_8Qixs

    if path.startswith("gs://"):
        audio = RecognitionAudio(uri=path)
    else:
        with open(path, "rb") as file:
            audio = RecognitionAudio(content=file.read())

    kwargs = {
        "audio_channel_count": 2,  # It fails otherwise for many audios. FIXME: fails with mono
    }

    if path.endswith(".opus"):
        kwargs["encoding"] = RecognitionConfig.AudioEncoding.OGG_OPUS  # All our Opus videos are in an Ogg container.
        # When using Ogg-Opus, the endpoint needs the following fields.
        # See https://cloud.google.com/speech-to-text/docs/encoding
        kwargs["sample_rate"] = 48000  # All our Opus audios I've seen use this rate (at least 100).
    else:
        kwargs["encoding"] = RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED

    metadata = RecognitionMetadata(original_media_type=RecognitionMetadata.OriginalMediaType.VIDEO)
    config = RecognitionConfig(language_code="en-US", enable_word_time_offsets=True, enable_word_confidence=True,
                               # Option not supported in the enhanced video model:
                               # alternative_language_codes=["en-GB", "en-IN", "en-AU"],
                               enable_automatic_punctuation=True, use_enhanced=True, model="video", metadata=metadata,
                               diarization_config=SpeakerDiarizationConfig(enable_speaker_diarization=True,
                                                                           min_speaker_count=1, max_speaker_count=10),
                               **kwargs)
    response = SpeechClient().long_running_recognize(config=config, audio=audio)
    result = response.result(timeout=10000)
    print(type(result).to_json(result))


if __name__ == "__main__":
    main()
