#!/bin/bash
sudo apt update -y
sudo apt install -y nano htop

pip3 install -q -r Hypercomplex-neural-networks/requirements.txt
#python3 ./Hypercomplex-neural-networks/Main.py