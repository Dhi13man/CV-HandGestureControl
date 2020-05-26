# CV-HandGestureControl
A Python based project to train a Machine Learning model to detect different hand shapes in real time with multi-threading, using Computer Vision, to control the PC.

---

# Usage:
Run main.py to use. Wait till "CALIBRATED" is shown.

It uses a K-Nearest Neighbors Model with 1 Nearest Neighbor, Support Vector Machine model with Linear kernel and a Random Forest Model with 55 estimators as classifiers in conjunction to make predictions.
The Mode of the three classifiers' predictions is considered the accurate prediction.

Interpretation when hand is in relevant area (marked by green box):
1. No Hand `0` 
2. High Five `1`
3. Middle Finger `2`
4. V Sign `3`

---

# Four Classification Models:

KNN Model (Optimum Nearest Neighbors = 5): 
--
1. Can detect 3 different Hand Gestures and lack thereof. Train Model for more.
2. Run `main.py` to use with Model stored in `KNN.pkl` to use.
3. Use Less number of images during training(with Detector(value) where value ~ 80) to resolve out of memory issue.

Support Vector Machine Model (Optimum Kernel = 'Linear'):
--
1. Can detect 3 different Hand Gestures and lack thereof. Train Model for more.
2. Run `main.py` to use with Model stored in `svm_linear.pkl` to use.
3. Use Less number of images during training(with Detector(value) where value ~ 80) to resolve out of memory issue.

Random Forest Model (Optimum number of Estimators = 55):
--
1. Can detect 3 different Hand Gestures and lack thereof. Train Model for more.
2. Run `main.py` to use with Model stored in `rf.pkl` to use.

Logistic Regression Model:
--
1. Perfectly predicts hand doing High Five gesture or lack thereof.
2. Run `main_logistic_regress.py` to use.
3. Model parameters stored in `lregression_parameters.npy`.


---
# Associated Scripts:
1. **`dataset_manips.py:`** Python script containing functions to **build new datasets, clear existing datasets, arrange existing dataset files** for more serialized naming etc in the `Datasets` folder.
2. **`directgameinp.py:`** Best solution to translate Models' predictions into useful input. Functions in it can be used to send input to games(or other applications) using functions like KeyDown() and KeyUp().\
  This can also be replaced with the `PyDirectInput` Python Library.\
**Note:** Input might still not be noticed by Games incorporating Direct Input protection. Haven't found any working alternative for them, other than programming a custom Keyboard Driver or Virtual Controller simulation with keyboard key binding. Feel free to suggest alternatives.
3. **`gesture_ML.py:`** Trains a KNN model and/or SVM model and/or Random Forest model based on the functions called with the image Datasets present in the different folders in the `Datasets` folder, enumerated based on the order they have been trained from. Then, it saves the resulting model as `KNN.pkl` and/or `svm_lin.pkl` and/or `rf.pkl`.
4. **`gesture_ML_logistic_regress.py:`** Trains a Logistic Regression model based on the image Datasets present in the different folders in the `Datasets` folder, enumerated based on the order they have been trained from. For best possible accuracy using Sigmoid, it only supports 2 possible classes enumerated as `0` and `1` based on order of training. Then, it saves the resulting model's parameters as `lregression_parameters.npy`. 
5. **`main.py:`** Incorporates pre-trained SVM, KNN and Random Forest Model together to detect **No hand `0`, High five `1`, Middle Finger `2`** and **Ok sign `3`** with OpenCV using device camera.
6. **`main_logistic_regress.py:`** Incorporates pre-trained Logistic Regression Model Parameters with Sigmoid function to detect **No hand `0`** or **Hand `1`** with OpenCV using device camera.
7. **`presskey.ahk or presskey.exe:`** Alternative solution to translate Models' predictions into useful input. It uses [AutoHotkey](https://www.autohotkey.com/docs/Tutorial.htm) scripting to accomplish this. [AutoHotkey](https://www.autohotkey.com/) may be downloaded to edit the .ahk script as needed. Then simply call the AutoHotkey script(with AutoHotkey installed)/executable from `main.py` or `main_logistic_regress.py`.\
 Using this is equivalent to simply using Python Libraries like `pynput`, `pyautogui` or `keyboard`, and that would be much simpler.\
**Note:** This method will fail in Almost all DirectX based and DirectInput Games and Applications.
8. **`visualizer.py:`** Heart of the software, called by all other scripts to isolate the hand from background through OpenCV contour detection using device camera, then use it to build dataset (which will then be used to train models) or classify gesture.

# Machine Learning Model Objects:
1. **`KNN.pkl:`** K-Nearest Neighbors Classifier Model Object stored as a Binary joblib Pickle Dump. Use `joblib.load` to load it into your scripts and use it's predict() method to classify 240x255x3 Black(0) and White(255) images into below mentioned classes.
2. **`lregression_parameters.npy:`** Contains the W and b parameters for a Sigmoid-based Logistic Regression model, to accurately predict whether a High Five gesture is present in a Black(0) and White(255) 240x255x3 Image. It has been stored as a Numpy save Dump using `numpy.save`. Use `numpy.load` with `allow_pickle=True` parameter to load the parameters into your scripts as a length 2 numpy array. Feed the resulting Linear equation formed from X as a suitable image into a Sigmoid function for classification
3. **`rf.pkl:`** Random Forest Classifier Model Object stored as a Binary joblib Pickle Dump. Use `joblib.load` to load it into your scripts and use it's predict() method to classify 240x255x3 Black(0) and White(255) images into below mentioned classes. Use `numpy.load` with `allow_pickle=True` parameter to load the parameters into your scripts as a length 2 numpy array. Feed the resulting Linear equation formed from X as a suitable image into a Sigmoid function for classification
4. **`svm_lin.pkl:`** Linear kernel Secure Vector Machine Model Object stored as a Binary joblib Pickle Dump. Use `joblib.load` to load it into your scripts and use it's predict() method to classify 240x255x3 Black(0) and White(255) images into below mentioned classes.
 
# Datasets:
All Image Datasets stored in the `Datasets` folder are self created using `dataset_manips.py`, incorporating `visualizer.py`. They currently contain 4 different types of Hand Gestures that are ready and to train models on:
1. No hand
2. High Five
3. Middle Finger
4. V Sign
5. Ok Sign (Not trained by models)

# Examples
The `Examples` folder contains two video examples of `main.py` in action in a Game and for Spotify song changing, all through key presses. 

---
# Working Demonstrations:
1. **[Changing songs in Spotify](https://github.com/Dhi13man/CV-HandGestureControl/blob/master/cvgesture.mp4)**
1. **Usage in Games (Game Used: [Orcs Must Die 2](https://store.steampowered.com/app/201790/Orcs_Must_Die_2/)):**\
[Substitue Mouse Input](https://github.com/Dhi13man/CV-HandGestureControl/tree/master/Examples/cvgesture1.mp4)\
[Substitute Keyboard and Mouse Input](https://github.com/Dhi13man/CV-HandGestureControl/tree/master/Examples/cvgesture2.mp4)
