# Data Preparation Steps  

The WikiLarge Train dataset was de-tokenized and feature values were calculated before splitting. The dataset was then randomly split into: two subsets of 1,000 sentences for evaluations, and the rest for training.  

Duplicate entries were removed from the training dataset. Filtering was applied by removing outliers and excluding instances where the target/source feature value ratio exceeded 2.0. After filtering, the dataset sizes were:  
- Train: 216,881 sentences  
- Eval Set-1: 755 sentences
- Eval set-2: 731 sentences  

To reduce OpenAI API costs, the test set was further limited to 200 sentences. The final dataset sizes were:  
- Train: 216,881 sentences  
- Eval set V1.0: 755 sentences
- Eval set V1.1: 200 sentences (used in our experiments as TEST set)
- Eval set V1.2: 531 sentences 
