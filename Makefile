mac: 
	python3 -m venv p38 --system-site-packages
	pip3 install -U pip
	pip3 install --upgrade -r requirements.txt

prepare-ubuntu:
	sudo apt-get install python3-venv

ubuntu:
	virtualenv --python=/usr/bin/python3.8 python38	
	. ./python38/bin/activate
	sudo pip3 install --upgrade -r requirements.txt
