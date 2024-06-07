# Hand-Sign-Detector

This project is all about lending a helping hand to those who communicate through hand signs because they are unable to speak. Picture this: someone who relies on hand gestures to express themselves, now equipped with a real-time hand gesture recognition system. Powered by machine learning and the magic of MediaPipe for hand landmark detection, this project lets users communicate effortlessly through their gestures.

Using just a webcam, the system detects and understands hand movements, translating them into meaningful commands or messages. Imagine being able to chat with someone using only your hands! This technology opens up a whole new world of communication for individuals who rely on hand signs.

In essence, this project is like a friendly companion for those who communicate without words. It's a tool that bridges the gap, making social interactions smoother and everyday conversations easier. With this system in place, everyone can join in, express themselves, and connect with others, regardless of their ability to speak.

## Installation

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [TensorFlow](https://www.tensorflow.org/)
You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).
    
## Run Locally

Clone the project

```bash
  git clone https://github.com/tanmoymondal-86/Hand-Sign-Detector
```

In a terminal or command window, navigate to the top-level project directory Hand-Sign-Detector/ (that contains this README) and run the following command:
```bash
  python sign_detector.py
```

## Datasets -

![0](./images/example_0.jpg?raw=true "0")
![1](./images/example_1.jpg?raw=true "1")
![2](./images/example_2.jpg?raw=true "2")
![3](./images/example_3.jpg?raw=true "3")
![4](./images/example_4.jpg?raw=true "4")
<br>

![5](./images/example_5.jpg?raw=true "5")
![6](./images/example_6.jpg?raw=true "6")
![7](./images/example_7.jpg?raw=true "7")
![8](./images/example_8.jpg?raw=true "8")
![9](./images/example_9.jpg?raw=true "9")



## To make a model on custom dataset:
- First run images.py to capture the dataset
```bash
  python images.py
```
- Then run mediapipe.ipynb
```bash
jupyter notebook mediapipe.ipynb
```
or just open the mediapipe.ipynb file in VS Code.

- A nums_detector.pickle file will be generated.

- Then run train_model.ipynb
```bash
jupyter notebook train_model.ipynb
```
- Finally run the following command:
```bash
python sign_detector.py
```
