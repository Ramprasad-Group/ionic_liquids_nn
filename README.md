# Ionic Liquids Neural Network for Conductivity


This repository contains the code to train the Ionic Liquids Neural Network for Conductivity, a deep neural network that has learned to predict conductivity (in S/m) of pure-compound, binary ionic liquids (ILs). It can be used to test the conductivity of tailored ILs rapidly and accurately using completely machine-driven property predictions. The paper containing relevant information about the dataset, the neural network, and the resulting observable trends can be found in the Journal of Chemical Physics.   

## Abstract: 
Ionic liquids (ILs) are salts, composed of asymmetric cations and anions, typically existing as liquids at ambient temperatures. They have found widespread applications in energy storage devices, dye-sensitized solar cells, and sensors because of their high ionic conductivity and inherent thermal stability. However, measuring the conductivity of ILs by physical methods is time-consuming and expensive, whereas the use of computational screening and testing methods can be rapid and effective. In this study, we used experimentally measured and published data to construct a deep neural network capable of making rapid and accurate predictions of the conductivity of ILs. The neural network is trained on 406 unique and chemically diverse ILs. This model is one of the most chemically diverse conductivity prediction models to date and improves on previous studies that are constrained by the availability of data, the environmental conditions, or the IL base. Feature engineering techniques were employed to identify key chemo-structural characteristics that correlate positively or negatively with the ionic conductivity. These features are capable of being used as guidelines to design and synthesize new highly conductive ILs. This work shows the potential for machine-learning models to accelerate the rate of identification and testing of tailored, high-conductivity ILs.

## Ionic Liquid Fingerprints

We recommend using Fingerprinting_RdKit_ILs.py for generating ionic liquid fingerprints. The script utilizes the RdKit.Chem python library.  To use the script, SMILES strings must be generated as identification for the ionic liquids, IUPAC names will not work. 

## NIST ILThermo Data Set

The data used to train the neural network is from the NIST ILThermo database. The final data set used (after filtering) is found in the Supplementary_Material_ILThermoDataset_ALL.csv file. This includes the IL fingerprint.

## Authors

Rohan Datta and Shruti Venkatram

- Paper: R. Datta, R. Ramprasad, and S. Venkatram, Conductivity prediction model for ionic liquids using machine learning, J. Chem. Phys. 156, 214505.
