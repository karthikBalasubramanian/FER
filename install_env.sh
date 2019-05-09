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
conda install scikit
conda install sklearn
conda install scikit-learn
conda install keras
conda install pandas
conda install opencv
brew install cmake
brew install boost
brew install boost-python3
ln -s /usr/local/lib/libboost_python36.a /usr/local/lib/libboost_python3.a 
ln -s /usr/local/lib/libboost_python36.dylib /usr/local/lib/libboost_python3.dylib 
pip install opencv-contrib-python
pip install dlib