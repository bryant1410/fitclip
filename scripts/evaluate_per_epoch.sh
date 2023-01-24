#!/usr/bin/env bash

set -x

for file in /mnt/localssd/tmp/epoch=*.ckpt; do
  ./scripts/prepare_trained_clip_checkpoint_for_evaluation.py "$file" a.pt
  python -m aligner \
    --multirun \
    command=evaluate \
    +encoder._target_=aligner.wise.wise \
      +encoder@encoder.model1=clip_vit_b_16 \
      +encoder@encoder.model2=clip_from_pretrained \
        +encoder.model2.model.name="$PWD"/a.pt \
      +encoder.weight_for_2=0.4 \
    data=moments_in_time,msrvtt,webvid,youcook2 \
    silent=true
#        +encoder.model2.model.name=<(./scripts/checkpoint_to_state_dict.py "$file") \
done
