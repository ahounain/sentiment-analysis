# Emotion Recognition from Speech: Multimodal Deep Learning Approach

# NOTE for grader : Various google colab ipynb files are in the src/notebooks folder
## Project Overview
This project implements a deep learning-based emotion recognition system using the RAVDESS dataset, leveraging both acoustic features and contextual metadata.

## Prerequisites
- Python 3.8+
- pip (Python package manager)

Create a virtual environment (optional)

# Install dependencies
pip install -r requirments.txt

Download the RAVDESS dataset from https://zenodo.org/records/1188976

Replace the path of DATASET_PATH in the main of src/training/train_emotion.py with the path to RAVDESS.

# Training 

python train_model.py 

# Running 


Once the model has been created go back to the src folder and run 

python main.py 

If you want to change the audio file, you can make one by running

python inputaudio.py 

which will overwrite audio.wav with a five second recording (you can tweak the code to make it longer) of your voice.


Running will take a while the first time because it needs to install Wav2Vec2 and MMS-LID. 
You will only have to install it once because after the fact it gets cached, once that's all done though
it still takes a while for the model to predict emotion depending on your processing power.

Once that's done it will then spit out the results to the terminal.