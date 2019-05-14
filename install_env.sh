# prerequisite have anaconda installed
# anaconda installation instructions - https://docs.anaconda.com/anaconda/install/
conda info --envs
conda create -n fer python=3.6
conda activate fer
conda install ipykernel
python -m ipykernel install --user --name=fer
conda update conda
conda install jupyter
conda install scipy
conda install scikit-learn
conda install keras
conda install pandas
conda install opencv
# sudo apt-get install cmake
brew install cmake
brew install boost
# https://askubuntu.com/questions/944035/installing-libboost-python-dev-for-python3-without-installing-python2-7
brew install boost-python3
ln -s /usr/local/lib/libboost_python36.a /usr/local/lib/libboost_python3.a 
ln -s /usr/local/lib/libboost_python36.dylib /usr/local/lib/libboost_python3.dylib 
pip install opencv-contrib-python
pip install dlib
conda install matplotlib
conda install tqdm
conda install pillow