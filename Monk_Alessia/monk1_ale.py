from ucimlrepo import fetch_ucirepo 
import numpy as np
  
# fetch dataset 
monk_s_problems = fetch_ucirepo(id=70) 
  
# data (as pandas dataframes) 
X = monk_s_problems.data.features 
y = monk_s_problems.data.targets 
  
# metadata 
print(monk_s_problems.metadata) 

# variable information 
print(monk_s_problems.variables) 

