# Learning over Sets for Databases 
This repo contains the code for the paper, "Learning over Sets for Databases".  

## Setting up Conda Environment

To get the required libraries, we provide a conda environment `environment.yml`.
To create the environment use the command:

`conda env create -f environment.yml`

To activate the environment use the command:

`conda activate clsm`


## Main Modules

- [`set_boundaries.py`](./set_boundaries.py): a learned set index or a learned cardinality estimator (param --indexing indicates index, the absence indicates cardinality)
- [`set_bf.py`](./set_bf.py): a learned bloom filter
- [`params.py`](./params.py): all the possible settings for the structures


## Quick Start
The execution is performed, by simple executing a specific file once the parameters in the params file, or the command line are defined.
The important parameters are:
- compression: True/False
- indexing: True/False, relevant only for the `set_boundaries.py`
- outlierremoval: Triggers the creation of a hybrid structure with defined percentage of removal, starting step of removal, and step of removal 



## Cardinality Estimation Task Experiments
Example execution:
```bash
# example training of the model without compression
python set_boundaries.py --training --epochs 100
# example training of the model with compression
python set_boundaries.py --training --compression --epochs 100
# example training of the model without compression and outlier removal
python set_boundaries.py --training --outlierremoval --startremoval 90 --stepremoval 40 --boundaryremoval 90 --epochs 100
# example training of the model with compression and outlier removal
python set_boundaries.py --training --compression --outlierremoval --startremoval 90 --stepremoval 40 --boundaryremoval 90 --epochs 100
# testing of the model without compression
python set_boundaries.py
# testing of the model with compression
python set_boundaries.py --compression
# testing of the model without compression and outlier removal
python set_boundaries.py --outlierremoval
# testing of the model with compression and outlier removal
python set_boundaries.py --compression --outlierremoval
```

To recreate the experiments in the paper for cardinality estimation, follow the following instructions:
For RW dataset, here noted as RW_modified, we use for encode, decode, middle neurons a setting of 64 and an embedding 32.
For SD and Tweets dataset, here noted as RW_modified, we use for encode, decode, middle neurons a setting of 128 and an embedding 32.
To set the parameters use the parameters as defined in the params file.



## Indexing Task Experiments
Example execution:
```bash
# example training of the model without compression
python set_boundaries.py --indexing --training --epochs 100
# example training of the model with compression
python set_boundaries.py --indexing --training --compression --epochs 100
# example training of the model without compression and outlier removal
python set_boundaries.py --indexing --training --outlierremoval --startremoval 90 --stepremoval 40 --boundaryremoval 90 --epochs 100
# example training of the model with compression and outlier removal
python set_boundaries.py --indexing --training --compression --outlierremoval --startremoval 90 --stepremoval 40 --boundaryremoval 90 --epochs 100
# testing of the model without compression
python set_boundaries.py --indexing
# testing of the model with compression
python set_boundaries.py --indexing --compression
# testing of the model without compression and outlier removal
python set_boundaries.py --indexing --outlierremoval
# testing of the model with compression and outlier removal
python set_boundaries.py --indexing --compression --outlierremoval
```

To recreate the experiments in the paper for cardinality estimation, follow the following instructions:
As the encode, decode, middle neurons a setting of 8 and an embedding 4. The outlier removal is as defined in the paper.

Note that for execution of the model, unlike for cardinality estimation, outlier removal is needed.
For the outlier removal we use the parameters:
```bash
--indexing --startremoval 40 --stepremoval 50 --boundaryremoval X --epochs 50
```
where X is 90 for RW-200k and RW-3M, 70 for SD and 60 for Tweets/hashtags3.


## Bloom Filter Task Experiments
Example execution:
```bash
# example training of the model without compression
python set_bf.py --training --epochs 100
# example training of the model with compression
python set_bf.py --training --compression --epochs 100
# testing of the model without compression
python set_bf.py
# testing of the model with compression
python set_bf.py --compression
```

To recreate the experiments in the paper for cardinality estimation, follow the following instructions:
As the encode, decode, middle neurons a setting of 8 and an embedding 2.
