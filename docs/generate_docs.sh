#!/bin/bash

cd ~/py-automl
python3 setup.py install
sphinx-apidoc -f -o docs/source pyautoml/
cd docs
make html
