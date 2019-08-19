#!/bin/bash

cd ~/py-automl
sphinx-apidoc -f -o docs/source pyautoml/
cd docs
make html
