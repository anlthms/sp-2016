### Description

Seizure Prediction

- [Competition page at Kaggle](https://www.kaggle.com/c/melbourne-university-seizure-prediction)
- This is a proof-of-concept for applying deep learning techniques to EEG data converted into spectrograms.
- This code builds a separate model for each subject (there are three subjects).

### Usage

These steps take about 30 minutes on a system with 4 processors and a single GPU. **Tested only on Ubuntu**.

1. Download and install neon 1.5.5

    ```
    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    git checkout v1.5.5
    make
    source .venv/bin/activate
    ```
2. Verify neon installation
    Make sure that no errors are printed from running this command:
    ```
    ./examples/cifar10_msra.py -e1
    ```

3. Install prerequisites

    ```
    pip install sklearn scikits.audiolab
    ```
4. Download the data files from [Kaggle](https://www.kaggle.com/c/melbourne-university-seizure-prediction/data):

    Save all files to a directory (we will refer to this directory as /path/to/data) and unzip the .zip files.

5. Clone this repository

    ```
    git clone https://github.com/anlthms/sp-2016.git
    cd sp-2016
    ```
6. Train models and generate predictions

    ```
    ./run.sh /path/to/data
    ```
7. Evaluate predictions

    Submit subm.csv to [Kaggle](https://www.kaggle.com/c/melbourne-university-seizure-prediction/submissions/attach)

### Notes
- The model requires 4GB of device memory.
- If using AWS, see slide 10 on [this deck] (https://github.com/anlthms/meetup2/blob/master/audio-pattern-recognition.pdf) for instructions on how to configure an EC2 instance.
- The first run takes longer due to conversion of .mat files into .wav files. Once the data is prepared, subsequent runs can complete in less than 15 minutes.
- Conversion of data to spectrograms is performed on the fly by neon.
- As provided, the run.sh script uses data from the first electrode. This means that only 1/16th of the data is used.
- A leaderboard AUC score of 0.56 may be obtained by using this code as is.
