# Data-Analysis-Term-Project

#### 윤혜정승아팀: 202CPG03 SeungA Chung, 200AIG01 HyeJung Yoon

This is a project for facial expression recognition based on CNN model.

1. Download the dataset from the below link:<br>
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

2. Download our code, especially the code in 'Final Code' folder, saved models in 'CNN_final_hdf5 with figures' folder, and haarcascade_frontalface_default.xml file

3. Before run the 'Facial Expression Recognition System Code', please check the detection_model_path and emotion_model_path.
Also, there are some requirements to run this code:
tensorflow, keras, imutils, opencv should be already installed.
If not, please install first by using 'pip install'.

+ If you are going to run the 'Facial Expression Recognition System Code' in GPU environment, please add the code below at the begins of the code.

<pre>
<code>
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
</code>
</pre>

# Responsibilities

#### SeungA Chung: 
- Experiment design
- Overall coding including the experiment and a system
- CNN structure experiment 
- Paper(Abstract, Introduction, Method(specifically implementation part), Experiment(specifically Model Structure experiment part), Suggestion for Facial Expression Recognition System, Conclusion)
- Document
- Demo

#### HyeJung Yoon: 
- Experiment design
- Epoch experiment
- Regularization experiment
- Paper(Introduction, Related Work, Method(specifically dataset and design of neural network part), Experiment(specifically Epoch experiment, Regularization experiment part))
- PPT
- Demo

