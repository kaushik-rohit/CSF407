#!/bin/bash

echo "Installing virtualenv"
pip install virtualenv 

echo "Creating a virtual environment named openai"
virtualenv ENV

echo "Entering Virtual ENV"
source ./ENV/bin/activate #activate the environment openai

echo "Python package installation about to start in 1.2.3"

echo "Installing OpenAI GYM"
pip install gym

echo "Installing some of the know game env in Open ai"
pip install gym['atari']


echo "Installing System dependencies for PLE. Will require Root access"
sudo apt-get install mercurial python-dev python-numpy python-opengl \
    libav-tools libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev \
    libsdl1.2-dev libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev \
    libtiff5-dev libx11-6 libx11-dev fluid-soundfont-gm timgm6mb-soundfont \
    xfonts-base xfonts-100dpi xfonts-75dpi xfonts-cyrillic fontconfig fonts-freefont-ttf

sudo apt-get install python-pygame

echo "Installing PyGame-Learning-Environment"
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip install -e .
cd ..

pip install ppaquette-gym-doom
pip install pygame

git clone https://github.com/lusob/gym-ple.git
cd gym-ple/
pip install -e .
cd ..

echo "Removing Unwanted files"
rm -rf gym-ple
rm -rf PyGame-Learning-Environment
