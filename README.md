# CODE FOR Conditional Generative Diffusion Enables Ultra-Fast Battery Health Estimation from Random 100mV Charging Fragments

This repository contains the code for conditional generative diffusion based battery health estimation from random 100 mV charging fragments.

The project is organized by battery chemistry:

```text
NCA/
NCM/
LFP/
```

Each subproject follows the same workflow:

```text
Data_process -> Diffusion -> soh
```

Please see the README files inside each subfolder for detailed instructions.

## Data

This GitHub repository includes code and selected test data. The training data are not included in the GitHub repository.

The same code and data package can be downloaded from Zenodo:

https://zenodo.org/records/18298930?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk3MDBiOWU4LTE3OTQtNDE1OS1hZmU1LTNjMmFhZWRmZGI4NSIsImRhdGEiOnt9LCJyYW5kb20iOiIxZTljYTlhODg2NzM0NWJjMzAzYWMyZTViNDc2MzkyYyJ9.Hx20U4EsSuo8xyKf2W88yd_anROP8-CZEoRtAFu4kxS07MouvonWWJ15lH9SnHPE8kr8qrqvZ7_-SC5FPsGR1A

## Quick Use

For the full workflow, run the steps in each chemistry folder:

```text
Data_process -> Diffusion -> soh
```

For quick testing with prepared SOH input files and checkpoints, enter the corresponding `soh` folder and run:

```bash
python save_npy.py
```

The `save_npy.py` script performs test-time SOH inference and saves per-battery prediction results.
