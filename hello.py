from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
monk_s_problems = fetch_ucirepo(id=70) 
  
# data (as pandas dataframes) 
X = monk_s_problems.data.features 
y = monk_s_problems.data.targets 
  
# metadata 

