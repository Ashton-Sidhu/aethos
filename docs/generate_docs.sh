#!/bin/bash

cd ~/aethos
python3 setup.py install
sphinx-apidoc -f -o docs/source aethos/
cd docs
make html
