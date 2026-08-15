# Inference code for the Sybil model


1. Install requirements
```
torch
lighting
hydra-core
numpy
tqdm
```

2. Download the checkpoints.tar.gz and extract to
```
./data/weights/sybil/checkpoints
```

3. Download the sample_from_nlst.nii.gz and put it in directory 
```
./data/ct
```

3. Run the testing inference
```
HYDRA_FULL_ERROR=1 python src/infere_sybil.py --config-name test_sybil_ensemble_inference
```

The output should be `tensor([[0.1423, 0.0040, 0.0157, 0.0277, 0.0384, 0.0506, 0.0741]])`