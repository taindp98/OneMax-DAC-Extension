# This folder contains the trained DDQN checkpoints and policies produced from other approaches
## DDQN checkpoints
There are 7 checkpoints corresponding to the total number of examined problem sizes.
```plaintext
resources/
├── ddqn_ckpt/                      
│   ├── best_model_shifting_n50.pt                    
│   ├── best_model_shifting_n100.pt                   
│   ├── best_model_shifting_n200.pt                    
│   ├── best_model_shifting_n300.pt                    
│   ├── best_model_shifting_n500.pt                    
│   ├── best_model_shifting_n1000.pt                    
│   └── best_model_shifting_n2000.pt                    
```
Please refer to this notebook [test.ipynb](../notebooks/test.ipynb) to replicate the ERT from these checkpoints.
## Other Approaches
```plaintext
resources/
├── other_methods/                      
│   ├── irace_cascading_policies.json                                  
│   └── optimal_policies.pt                    
```
We compare our RL-based DAC with two approaches:
1. IRACE cascading tuning approach [(Chen et al., 2023)](#reference) for 3 problem sizes `n=[500, 1000, 2000]` following the format
```json
{
    "500": {            ## problem size
        "bin1": [],     ## each bin contains a policy
        "bin2": []
    }
}
```

2. Optimal policy [(Chen et al., 2023)](#reference) containing full settings `n=[50, 100, 200, 300, 500, 1000, 2000]`.
```json
{
    "50": [],       ## contains a policy
    "100": [],
}
```

## Reference
```plaintext
@inproceedings{chen2023using,
  title={Using automated algorithm configuration for parameter control},
  author={Chen, Deyao and Buzdalov, Maxim and Doerr, Carola and Dang, Nguyen},
  booktitle={Proceedings of the 17th ACM/SIGEVO Conference on Foundations of Genetic Algorithms},
  pages={38--49},
  year={2023}
}
```