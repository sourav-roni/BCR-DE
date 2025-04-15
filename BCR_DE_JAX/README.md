# BCR-DE-JAX
This folder is the JAX implementation version of BCR-DE

# Set up environment

Please install JAX, Flax, Optax 
```Bash
pip install -U "jax[cuda12]"
```
```Bash
pip install flax
```
```Bash
pip install optax
```

# Regression
The main file is [regression.py](./regression.py) and it is used to perform experiments to predict Heart Rate(HR), Respiratory Rate (RR) and Oxygen Saturation level (SpO2). Below is a sample command to run one such experiment. Note that chaning the configs will run the different variants of RR, HR and SpO2.

```
python regression.py --config_path ./configs/regression/RR.json
```

# Classification
The main file for performing classification of Eigenworms is present in [classification.py](./classification.py). The two different configs provided in this case will make use of BCR_DE and noisy version of BCR_DE introduced in the paper. Sample command:
```
python classification.py --config_path ./configs/classification/eigenworm.json
```

# Autoencoding
Main file for autoencoding time series sequence of ECG/PPG is present in [autoencode.py](./autoencode.py)
```
python autoencode.py --dataset_type ECG
```

# Denoising autoencoding
For denoising autoencoding the main file is [denoise_AE.py](./denoise_AE.py). One can control the noise sensitivity from the parameters for both ECG and PPG sequence.
```
python denoise_AE.py --dataset_type ECG
```

# Masked autoencoding
In order to perform masked reconstruction, the main file is [masked_AE.py](./masked_AE.py). As for other sequence to sequence tasks presented here, there is a choice for ECG/PPG data. 
```
python masked_AE.py --dataset_type ECG
```

# Coupled Differential Equation
One of the key capabilities of BCR_DE as highlighted in the paper is that of modelling the dynamics in a coupled differntial equation system. The main file for this case is [coupled.py](./coupled.py). We present several such scenarios, the configurations of which are provided and can be used as follows:
```
python coupled.py --config_path ./configs/coupled_diffeq/HH.json
```

Please note that use of GPU is advised to achieve significant speedup in computation. However, the code also can be run on CPU.

# Note
- Regarding saving checkpoints in Flax, we have encountered issues with the Orbrax integration when attempting to save a Flax model. The Orbrax team is actively working to resolve this, and we will update the version as soon as the issue has been addressed.
