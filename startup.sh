#!/bin/bash

git clone https://github.com/dkwapisz/Hypercomplex-neural-networks.git

sudo apt update -y
sudo apt install -y nano htop

pip3 install -q -r Hypercomplex-neural-networks/requirements.txt
#python3 ./Hypercomplex-neural-networks/Main.py