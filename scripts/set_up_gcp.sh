#!/usr/bin/env bash

set -ex

sudo apt update
sudo apt install -y --no-install-recommends \
  awscli \
  mdadm \
  unattended-upgrades

sudo systemctl enable unattended-upgrades.service
sudo systemctl start unattended-upgrades.service

sudo mv /lib/systemd/system/nvidia-persistenced.service{,.bak}
cat <<EOF | sudo tee /lib/systemd/system/nvidia-persistenced.service > /dev/null
[Unit]
Description=NVIDIA Persistence Daemon
Wants=syslog.target

[Service]
Type=forking
PIDFile=/var/run/nvidia-persistenced/nvidia-persistenced.pid
Restart=always
ExecStart=/usr/bin/nvidia-persistenced --verbose
ExecStopPost=/bin/rm -rf /var/run/nvidia persistenced

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl unmask nvidia-persistenced
sudo systemctl enable nvidia-persistenced
sudo systemctl start nvidia-persistenced

eval "$(conda shell.bash hook)"
conda init

conda update -y conda

conda install -y conda-forge::mamba

mamba install -y pip
pip install gpustat

mkdir ~/.aws
cat <<EOF > ~/.aws/credentials
[default]
EOF

cat <<EOF . ~/.bashrc
alias aligner='python -m aligner'

alias evaluate='python -m aligner command=evaluate'
alias train='python -m aligner command=train'
EOF

SCRATCH_DIR="/scratch"
sudo mkdir -p "$SCRATCH_DIR"

ln -s $SCRATCH_DIR/cache_cp ~/.cache/cp

# Run the following commands each time the local SSDs are wiped out for whatever reason.
# (first set SCRATCH_DIR env var again).

sudo mdadm --create /dev/md0 --level=0 --raid-devices=8 /dev/nvme0n{1..8}
sudo mkfs.ext4 -F /dev/md0
sudo mount /dev/md0 "$SCRATCH_DIR"
sudo chmod a+w "$SCRATCH_DIR"

mkdir $SCRATCH_DIR/cache_cp
