PYTHON=python3

.PHONY: install test run run-null run-real clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install numpy pandas matplotlib scikit-learn pytest

test:
	$(PYTHON) -m pytest tests

run:
	$(PYTHON) sportka/experiments/run_experiment_v2.py

# run-null is an alias for run; both use synthetic data when no --csv is given
run-null:
	$(PYTHON) sportka/experiments/run_experiment_v2.py

run-real:
	$(PYTHON) sportka/experiments/run_experiment_v2.py --csv data/sportka.csv

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__

mac:
	python3 -m venv p312 --system-site-packages
	pip install -U pip
	pip install --upgrade -r requirements.txt

prepare-ubuntu:
	sudo apt-get install python3-venv

ubuntu:
	pip3 install --user virtualenv
	virtualenv --python=/usr/bin/python3.8 python38
	. ./python38/bin/activate
	sudo pip install --upgrade --ignore-installed  -r requirements.txt
