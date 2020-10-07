mac: 
	python3 -m venv p38 --system-site-packages
	pip install -U pip
	pip install --upgrade -r requirements.txt

prepare-ubuntu:
	sudo apt-get install python3-venv

ubuntu:
	#pip3 install --user virtualenv
	virtualenv --python=/usr/bin/python3.8 python38	
	. ./python38/bin/activate
	sudo pip install --upgrade --ignore-installed  -r requirements.txt
