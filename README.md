# FitCLIP

This repo contains the code for the BMVC 2022 paper [FitCLIP: Refining Large-Scale Pretrained Image-Text Models for
Zero-Shot Video Understanding Tasks](https://bmvc2022.mpi-inf.mpg.de/939/).

## Setup

Having Conda installed:

```bash
conda env create
conda activate sm
```

### Download the datasets and models

To use many of the datasets we used here, you need to download them. Go to their official website to find how to
download each of them. Check out the config files under `config/data` to find out what paths you need to set up for
them.

Similarly, to use many of the pre-trained models, you may need to download them and place them under a specific path (or
change the path). Check out the configs under `config/encoder`. You may need to preprocess them as well so to only have
the state dict (as opposed to the whole checkpoint, including for example the optimizer state). Checkout the scripts
under `scripts/` to preprocess them.

## Run the evaluation

Run like:

```bash
python -m aligner command=evaluate encoder=$MODEL data=$DATASET
```

Checkout the options with `--help` and the available configs under `config/`. Next are some example runs.

### CLIP on WebVid val

Run like:

```bash
python -m aligner command=evaluate encoder=clip_vit_b_16 data=webvid
```

### Frozen in Time on WebVid val

```bash
python -m aligner command=evaluate encoder=frozen_in_time data=webvid
```

### Evaluate on multiple benchmarks at the same time

```bash
python -m aligner --multirun command=evaluate encoder=clip_vit_b_16 \
  data=didemo,moments_in_time,msrvtt,ucf101,webvid,youcook2
```

### Evaluate a custom checkpoint

Suppose the checkpoint path is `a.pt`. Then, run:

```bash
python -m aligner \
    --multirun \
    command=evaluate \
    encoder=clip_from_pretrained \
    +encoder.model.name=$PWD/a.pt \
    data=moments_in_time,msrvtt,webvid,youcook2 \
    silent=true
```

## Save a model's predictions

```bash
python -m aligner command=predict
```

It'll be saved in `predictions.pt`.

You can see the options with `--help` and change the config file accordingly.

## Train a model (reproduce the paper results)

Run:

```bash
python -m aligner \
  --config-name teacher_student_train.yaml \
  command=train \
  +encoder@encoder.student=clip_vit_b_16 \
  +encoder@encoder.teacher=clip_vit_b_16 \
  data=mixed_batch_webvid_4_5k_all \
  ++model.fit_temperature=false \
  ++trainer.val_check_interval=30 \
  ++trainer.callbacks.3.train_time_interval.hours=0 \
  ++trainer.callbacks.3.train_time_interval.seconds=30 \
  ++trainer.callbacks.3.save_top_k=-1 \
  ++model.labeled_dataset_loss_share=0.9999
```

Then, grab the latest checkpoint generated under `ckpt=outputs/${DATE_AND_TIME}/checkpoints/best_labeled.ckpt`, and get
the student model:

```bash
./scripts/checkpoint_to_state_dict.py "$ckpt" > $student
```

Then you can evaluate it:

```bash
aligner \
  --multirun \
  command=evaluate \
  encoder=wise \
    +encoder@encoder.model1=clip_vit_b_16 \
    +encoder@encoder.model2=clip_from_pretrained \
      +encoder.model2.model.name="$student" \
  data=didemo,moments_in_time,msrvtt,ucf101,webvid,youcook2 \
  silent=true
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{Castro_2022_BMVC,
    author    = {Santiago Castro and Fabian Caba},
    title     = {{FitCLIP}: Refining Large-Scale Pretrained Image-Text Models for Zero-Shot Video Understanding Tasks},
    booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
    publisher = {{BMVA} Press},
    year      = {2022},
    url       = {https://bmvc2022.mpi-inf.mpg.de/0939.pdf}
}
```

## Troubleshooting

### Hydra shell completion doesn't work

See https://github.com/facebookresearch/hydra/issues/1957

### UCF101 website SSL certificate is not recognized

[The problem is that the server's certificate chain is
incomplete](https://www.ssllabs.com/ssltest/analyze.html?d=www.crcv.ucf.edu). The intermediate CA cert can be manually 
added:

```bash
sudo sh -c "curl https://www.incommon.org/custom/certificates/repository/sha384%20Intermediate%20cert.txt \
  > /usr/local/share/ca-certificates/incommon.crt"
sudo update-ca-certificates
# The requests library CA cert list also needs to be updated. Run like:
curl https://www.incommon.org/custom/certificates/repository/sha384%20Intermediate%20cert.txt \
  >> $CONDA_PREFIX/lib/python3.8/site-packages/certifi/cacert.pem
```

### Protobuf version error

If you have an error like:

> This program was compiled against version 3.9.2 of the Protocol Buffer runtime
> library, which is not compatible with the installed version (3.19.4).

Do:

```bash
conda install protobuf=3.9.2
```
