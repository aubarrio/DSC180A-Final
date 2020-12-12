# DSC180A-Final

How to run the run.py file.
This file reads in a dataset and runs our implementation of either GCN-LPA or GraphSage

## Datasets
* Cora Dataset: Network connection between publications
* OGB-arxviv: Classification of type of CS papers

## In order to run run.py through termal you must use: 
* `python run.py <file path> model aggregator`
* Aggregator only applies towards graphSage with the choices being: mean or max


If you were to run this on the cora dataset with the GCN-LPA model then you would run the command
* `python run.py data/cora gcn_lpa`

Examples for the other two models are as follows
* `python run.py data/cora graphsage mean`
* `python run.py data/cora gcn`

Simply running the command
              `python run.py`
would default to using the cora dataset and gcn model gathering its params from the config/params.json file

## Installation
In order to replicate the results found in our replication project we recommend following the steps listed below in order to make sure you meet the requirements to run our models.

#### Pip install
The recommended way to install and meet the requirements for our project we suggest using pip like so:
```bash
pip install -r requirements.txt
```
This uses the requirements.txt file we have set up in order to run this project


### Responsibilities
* Austin Le - Added functionality and dataparsing for OGB dataset as well as work on the Replication Report
* Aurelio Barrios - Implemented given replication models as well as work on the Replication Report
