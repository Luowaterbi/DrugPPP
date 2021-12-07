
# Drug PPP: Prediction of Pharmacokinetic Properties (Our Experiments Ver.)

## Get Started:

### Install Requirements 
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install pandas
conda install -c dglteam dgl-cuda10.1
conda install -c conda-forge rdkit
pip install transformers
pip install sklearn
```

### Run!
```
mkdir log
```

#### Run our model: SAIGN
```
source ./scripts/run_new_saign.sh 0
```
> Tips: number arg 0 is the gpu id

#### Run our old model based on CIGIN
```
source ./scripts/run_saign.sh 0
```

#### Run baseline model: CIGIN
```
source ./scripts/run_cigin.sh 0
```
