# This folder contains the trained DDQN checkpoints and policies produced from other approaches
## DDQN checkpoints
There are 7 checkpoints corresponding to the total number of examined problem sizes.
```plaintext
resources/
├── ddqn_ckpts/                      
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
1. IRACE cascading tuning approach [(Chen et al., 2023)](https://dl.acm.org/doi/abs/10.1145/3594805.3607127) for 3 problem sizes `n=[500, 1000, 2000]` following the format
```json
{
    "500": {            ## problem size
        "bin1": [],     ## each bin contains a policy
        "bin2": []
    }
}
```

2. Optimal policy [(Chen et al., 2023)](https://dl.acm.org/doi/abs/10.1145/3594805.3607127) containing full settings `n=[50, 100, 200, 300, 500, 1000, 2000]`.
```json
{
    "50": [],       ## contains a policy
    "100": [],
}
```
To test the performance of the optimal policy, we suggest a Python implementation like this:
```python
## policy: list of lambdas
## n: is the problem size
for i in range(n):
    cont_lbd = policy[i]
    floor_lbd = np.floor(cont_lbd)
    lbd = floor_lbd if cont_lbd - floor_lbd < 0.5 else floor_lbd + 1
    lbd = int(lbd)
    mutation_rate = lbd / n
    crossover_rate = 1 / lbd
```