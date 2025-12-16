# necessary installation

For the classification file

pip install opencv-python numpy matplotlib scipy scikit-learn torch torchvision joblib Pillow python-chess ipython


For the training file

pip install torch torchvision scikit-learn numpy matplotlib seaborn tqdm joblib


# this project is composed of two notebooks : 

training part, were we train an svm to classify images of chess pieces from the data folder and saves the models in the models folder.

classification part were we take an image of a chess board and rebuild the game using computer vision techniques and the saved classification model.


the classification model runs very well on the green/white boards provided in the boards folder. because we included only data related to that board. we could have added data and generalised the model to otehr boards

