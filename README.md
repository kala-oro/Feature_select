# Feature_select:
<p>
Complete the integration of some basic methods for feature filtering
<p>

# parameter:
    1. data source: you can choose one from house, iris, cancer 
    2. method_choise: 
       method_choise == 2 if you solve the univariate features selection assignment; 
            Part2.1 Pearson Correlation
            Part2.2 Mutual information and maximal information coefficient (MIC)
            Part2.3 Distance correlation 
            Part2.4 Model based ranking
        method_choise == 3 if you want to use linear model and regex method to select features;
            Part3.1 LinearRegression
            Part3.2 'L1' regex/ Lasso
            Part3.3 'L2' regex / Ridge
        mehtod_choise == 4 if use the RandomForest
            Part4.1 MDI(mean decrease impyrity)
            Part4.2 MDA(mean descrease accuracy)
        method——choise == 5 if use the Stability_selection. 
            Part5.1 RandomizedLasso
            Part5.2 Recursive feature eleimination RFE
# Method Detail:
    method_name2  2: 'pearson'  'MIC'  'Distance'  'Model_based'
    method_name3  3: 'linear'   'lasso'   'ridge'
    method_name4  4: 'MDI'  'MDA' 
    method_name5  5: 'Stab_sel'  'RFE'
# Result:
    for example: 
    {data_name = 'house', method_choise = 2, method_name2 = 'Model_based'}
| Index | FeatrueImportance|
| :-----| :--------------- |
| Feature_12 | 0.655959 | 
| Feature_5  | 0.510985 | 
| Feature_4  | 0.406344 | 
| Feature_2  | 0.322946 | 
| Feature_10 | 0.260156 |
| Feature_9  | 0.253293 | 
| Feature_0  | 0.168454 |
| Feature_1  | 0.168454 |
| Feature_8  | 0.15307  | 
| Feature_11 | 0.0863354|
| Feature_6  | 0.0176957|
| Feature_7  | 0.0107577|
| Feature_3  | -0.004984|
