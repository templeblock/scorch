#!/usr/bin/env bash

mkdir -p deps
cd deps
git clone https://github.com/torch/torch7.git
git clone https://github.com/torch/nn.git
git clone https://github.com/torch/cutorch.git
git clone https://github.com/torch/cunn.git
cd ..

