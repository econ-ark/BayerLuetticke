#!/bin/bash

python -m pip install -r requirements.txt

cd notebooks
ipython OneAsset-KS.py
ipython OneAsset-HANK.py
ipython TwoAsset-HANK.py
