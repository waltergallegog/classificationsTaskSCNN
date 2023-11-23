# <b>Impact of Encoding Techniques on the Classification of Raw Time-Variant Signals with Spiking Neural Networks</b>
## <b>Background</b>
Spiking Neural Networks (SNNs) represent the third generation of neural networks, whose technology is inspired by the human brain. Preliminary analysis carried out in state of the art neuromorphic works, highlight a number of advantages in the use of these technologies compared to the modern techniques used for data analytics. Since the information is processed in the form of an electrical impulse also referred to as spike, and by exploiting a sparsely connected neural structure and asynchronous processing of the neurons, these networks promise to be the solution to the enormous energy consumption of the ANNs, while maintaining competitive accuracy. Therefore, proceeding with a complete analysis of a machine learning pipeline on the analysis of sensor samples, I analyzed the preprocessing, coding, modeling and classification steps in a neuromorphic key.

As first I selected two type of dataset of time-varying signals, to be encoded in the spike domain, the Free Spoken Digit (FSD) and the WISDM. By carrying out an analysis of the types of data processed, I outlined a first classification, based on the organization of the information, identifying the temporal, spatial or a combination of the two, defining the classes of Temporal data, Spatial Data and Spatial-Temporal Data. This allowed me to identify the characteristics of a signal in order to choose the processing technique.

Referring to bio-inspired techniques, I was able to identify a plethora of pre-processing and coding techniques capable of encoding audio, video, olfactory, tactile signals, etc., which can be divided into classes, such as the Rate Encoding class where the information is encoded in the number of spikes per unit of time, and the Temporal Encoding where the information is included in the number of spikes, the interval between two spikes, the spike time, etc. By applying these techniques on temporal and spatial data, I identified specific criteria for selecting the most suitable technique.

Then as third phase I implemented a SNN construction and training process, based on the Transfer Learning method, performed by training an artificial neural network ANN and then inject the trained weights in a SNN twin. Specifically, in this work, this procedure was adopted to switch from a convolutional neural network CNN to a spiking CNN. The data to train the CNN architecture, achieved through the production of the sonogram, a reprocessing of the spike data converted into an image. Finally, the last step implemented is the encoding the sonogram through the rate coding, in order to make the information suitable for SNN processing. This method might seem laborious, however these steps allowed me to define a unique evaluation criterion of the various coding techniques. Observing test accuracy sets, 98\% accuracy was recorded for some types of encoding algorithms, while for others up to 8\%, this confirms that some encoding techniques are specifically implementable for Temporal data while others on Spatial data.

Once the best performing techniques were identified, I decided to conduct further investigations on synapse reduction techniques, which allow to reduce the size of the network making it suitable for use on embedded devices, thanks to a reduced memory footprint and a smaller computational cost. In some cases I have found that the accuracy of the reduced network exceeds the classification capabilities compared to the complete one, as in the case of the WISDM for which I observed an increase in accuracy starting from 86.7% up to 95.0%


### Reference
The reference material used to develop the research is obtained from:

- Riccardo Pignari, Gianvito Urgese, Vittorio Fra, Evelina Forno. "Impact of Encoding Techniques on the Classification of Raw Time-Variant Signals with Spiking Neural Networks" https://webthesis.biblio.polito.it/22859/
- Forno, E., Fra, V., Pignari, R., Macii, E., & Urgese, G. (2022). Spike encoding techniques for IoT time-varying signals benchmarked on a neuromorphic classification task. Frontiers in Neuroscience, 16, 999029. https://www.frontiersin.org/articles/10.3389/fnins.2022.999029/full


## Virtual environment configuration

The project makes use of the virtual environment creator conda.

Clone git repo:
```bash
git clone https://github.com/riccardopignari/classificationsTaskSCNN.git
```

miniconda installation (you can also use other conda flavors, like micromamba):
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
./Miniconda3-py38_4.10.3-Linux-x86_64.sh
```
To create the virtual environment and install all the necessary packages, run the commands:

```bash
conda create --name nest python=3.9.16
conda activate nest

# Installing package via CONDA
conda install pip
conda install -c conda-forge nest-simulator==2.20.1

# Installing package via PIP
pip install pynn==0.9.6
pip install tensorflow
pip install matplotlib
```

The commands above will create a virtual environment called `nest` and install all the necessary packages for the project. Keep in mind this process can be **SLOW** as the environment required for nest complicated and conda takes some time to solve it.

Alternatively, if you are on Linux 64, you can use the **solved** environment:

```bash
conda create --name nest --file solved_env_linux.txt
```

Two scripts are provided to create the environment on Linux and Windows:
- `automatic_conda_linux.sh`
- `automatic_conda_windows.sh`

If the environment creation from scratch is taking too long, you may want to try using micromamba instead:
https://github.com/mamba-org/mamba

micromamba is a drop-in replacement for conda that is much faster and solves environments in a fraction of the time, as it is implemented in C++.



## Project Structure
- `classificationFSD`, `classificationHAR`: the following folders contain all the scripts for the reproducibility of the experiments according to the development level:
  - `0-preliminaryAnalysis`;
  - `1-encoding`;
  - `2-trainingCNN`;
  - `3-inferenceSCNN`.
- `datasets`: folder containing all the raw level samples necessary for the creation of datasets for training in CNN and inference in SCNN;
  - Free Spoken Digit Dataset: https://github.com/Jakobovski/free-spoken-digit-dataset
  - Human Activity Recognition Dataset: https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset
- `networkModels`: destination folder for storing CNN networks keras models;
- `networkPerformance`: destination folder for all CNN and SCNN performance data;
- `utils`: It contains all the scripts needed for sample preprocessing and encoding. Definition of CNN networks and conversion to SCNN.


## Contact
- riccardo.pignari@polito.it
- gianvito.urgese@polito.it
- vittorio.fra@polito.it
- evelina.forno@polito.it
