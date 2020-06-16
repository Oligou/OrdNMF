This folder contains: 

- data: contains MovieLens (ML) and Taste Profile (TPS) datasets and preprocessing.
- function: 
	* preprocess_data.py used to split datasets into train set + test set. 
	- rec_eval used to calculate evaluation metrics.
- model: 
	* OrdNMF implementation described in the paper. 
	* dcPF implementation as described in Gouvert et al. (2019) (PF as a limit case)
- script: contains the experiments for both datasets
	* train: training models 
	* score: evaluating on test set
	* ppc: predictive posterior checks