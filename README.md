# DSC180A-Final

How to run the run.py file.
This file reads in a dataset and runs our implementation of either GCN-LPA or GraphSage

## 2 Datasets
* Cora Dataset: Network connection between publications
* OGB-arxviv: Classification of type of CS papers

## In order to run run.py through termal you must use: 
* `python run.py <file path> source`

If you were to run this on the twitch dataset then you would run the command
* `python run.py data/raw/twitch twitch`

Examples for the other two datasets are as follows
* `python run.py data/raw/cora cora`
* `python run.py data/raw/facebook facebook`

Simply running the command
              `python run.py`
would default to using the cora dataset gathering its params from the config/params.json file


### Responsibilities
* Austin Le - Added functionality and dataparsing for OGB dataset as well as work on the Replication Report
* Aurelio Barrios - Implemented given replication models as well as work on the Replication Report
