import pandas as pd
import numpy as np
import os
from env import get_connection
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
scaler= MinMaxScaler()



def plot_box(train):
    '''
    plots boxplots of continoues variables using the train dataset
    '''
    
    # List of columns
    cols= ['bedrooms', 'bathrooms', 'sqr_feet', 'property_value',]

    plt.figure(figsize=(16, 3))

    for x, col in enumerate(cols):

        plot_number= x + 1 

        # subplot.
        plt.subplot(1, len(cols), plot_number)

        
        # Title name using the column names.
        plt.title(col)

        # boxplot.
        sns.boxplot(data=train[[col]])

        # turns off the grid lines.
        plt.grid(False)

        # proper plot spacing
        plt.tight_layout()

    plt.show()
    
    
def plot_heatmap(corr):
    
    """
    plotting a heatmap using the correlation matrix 
    """

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr, cmap='Blues', annot=True, mask= mask)
    plt.title('Correlation Heat Map')
    plt.show()

    
def sqr_property(train):
    """
    Using joint plot to show Property Value and square feet
    """
    sns.jointplot(x="sqr_feet", y="property_value", data=train,  kind='reg', height=5, color='red',
              joint_kws={'color':'blue'},line_kws={"color": "red"})
    plt.xlabel('Total Square Feet')
    plt.ylabel('Total Property Value')
    plt.show()
    
def bed_property():
    sns.lmplot(data= train, x= 'bedrooms', y= 'property_value', scatter=True, line_kws={"color": "red"})
    plt.title('Correlation Between Property Value and Bedrooms')
    plt.xlabel('Total Bedrooms')
    plt.ylabel('Total Property Value')
    plt.show()
    
def year_property():
    sns.lmplot(data= train, x= 'year_built', y= 'property_value', scatter=True, line_kws={"color": "red"})
    plt.title('Correlation of Property Value and Year Built')
    plt.xlabel('Year Built')
    plt.ylabel('Total Property Value')
    plt.show()
    
def bath_property():  
    sns.lmplot(data= train, x= 'bathrooms', y= 'property_value',scatter=True, line_kws={"color": "red"}) 
    plt.title('Correlation Between Property Value and Bathrooms')
    plt.xlabel('Total Bathrooms')
    plt.ylabel('Total Property Value')
    plt.show()
    
def calc_baseline(y_train,y_val):
       
    y_train= pd.DataFrame(y_train)
    y_val= pd.DataFrame(y_val)

    # 1. Predict value_pred_mean
    value_pred_mean= y_train['property_value'].mean()
    y_train['value_pred_mean']= value_pred_mean
    y_val['value_pred_mean']= value_pred_mean

    # 2. compute value_pred_median
    value_pred_median = y_train['property_value'].median()
    y_train['value_pred_median']= value_pred_median
    y_val['value_pred_median']= value_pred_median

    # 3. RMSE of value_pred_mean
    rmse_train = mean_squared_error(y_train.property_value, y_train.value_pred_mean)**(1/2)
    rmse_val = mean_squared_error(y_val.property_value, y_val.value_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
        "\nValidate/Out-of-Sample: ", round(rmse_val, 2))

    # 4. RMSE of value_pred_median
    rmse_train = mean_squared_error(y_train.property_value, y_train.value_pred_median)**(1/2)
    rmse_val = mean_squared_error(y_val.property_value, y_val.value_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
        "\nValidate/Out-of-Sample: ", round(rmse_val, 2))
        
        
        


def six_split(train, val, test):
    """
    this functions splits the data into 6 different datasets. We will use them for 
    our modeling 
    """
    # split into X and y train dataset 
    X_train= train.drop(columns=['property_value'])
    y_train= train['property_value']

    # split into X and y val dataset 
    X_val= val.drop(columns=['property_value'])
    y_val= val['property_value']

    # split into X and y test dataset 
    X_test= test.drop(columns=['property_value'])
    y_test= test['property_value']

    y_train= pd.DataFrame(y_train)
    y_val= pd.DataFrame(y_val)
    y_test= pd.DataFrame(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def mmscale(X_train, X_val, X_test):
    """
    takes in 3 datasets and scales each one of them
    """
    
    # create the object
    scaler= MinMaxScaler(copy=True).fit(X_train)

    X_train_scaled= pd.DataFrame(scaler.transform(X_train), columns= X_train.columns.values).set_index([X_train.index.values])
    X_val_scaled= pd.DataFrame(scaler.transform(X_val), columns= X_val.columns.values).set_index([X_val.index.values])
    X_test_scaled= pd.DataFrame(scaler.transform(X_test), columns= X_test.columns.values).set_index([X_test.index.values])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    

    return X_train_scaled, X_val_scaled, X_test_scaled

def take_the_L(X_train_scaled, y_train,X_val_scaled, y_val):
    '''
    returns results for different model types for train and validate dataset
    '''
    pred_mean= y_train.property_value.mean()
    y_train['pred_mean']= pred_mean
    y_val['pred_mean']= pred_mean
    rmse_train = mean_squared_error(y_train.property_value, y_train.pred_mean, squared= False)
    rmse_val = mean_squared_error(y_val.property_value, y_val.pred_mean, squared= False)

    # save the results
    metric_df= pd.DataFrame(data= [{
        'model': 'baseline_mean',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.property_value, y_train.pred_mean),
        'rmse_validate': rmse_val,
        'r2_validate': explained_variance_score(y_val.property_value, y_val.pred_mean)
        }])

    #Linear Regression model
    # run the model
    lm= LinearRegression(normalize= True)
    lm.fit(X_train_scaled, y_train.property_value)
    y_train['pred_lm']= lm.predict(X_train_scaled)
    rmse_train = mean_squared_error(y_train.property_value, y_train.pred_lm)**(1/2)
    y_val['pred_lm']= lm.predict(X_val_scaled)
    rmse_val= mean_squared_error(y_val.property_value, y_val.pred_lm)**(1/2)

    # save the results
    metric_df= metric_df.append({
        'model': 'Linear Regression',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.property_value, y_train.pred_lm),
        'rmse_validate': rmse_val,
        'r2_validate': explained_variance_score(y_val.property_value, y_val.pred_lm)}, ignore_index= True)


    # LassoLars Model
    lars= LassoLars(alpha= 3)
    lars.fit(X_train_scaled, y_train.property_value)
    y_train['pred_lars']= lars.predict(X_train_scaled)
    rmse_train= mean_squared_error(y_train.property_value, y_train.pred_lars, squared= False)
    y_val['pred_lars']= lars.predict(X_val_scaled)
    rmse_val= mean_squared_error(y_val.property_value, y_val.pred_lars, squared= False)

    # save the results
    metric_df= metric_df.append({
        'model': 'LarsLasso, alpha 4',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.property_value, y_train.pred_lars),
        'rmse_validate': rmse_val,
        'r2_validate': explained_variance_score(y_val.property_value, y_val.pred_lars)}, ignore_index= True)

    # create the model object
    glm= TweedieRegressor(power= 0, alpha= 0)
    glm.fit(X_train_scaled, y_train.property_value)
    y_train['glm_pred']= glm.predict(X_train_scaled)
    rmse_train= mean_squared_error(y_train.property_value, y_train.glm_pred)**(1/2)
    y_val['glm_pred']= glm.predict(X_val_scaled)
    rmse_val= mean_squared_error(y_val.property_value, y_val.glm_pred)**(1/2)


    # save the results
    metric_df= metric_df.append({
        'model': 'Tweedie Regressor',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.property_value, y_train.glm_pred),
        'rmse_validate': rmse_val,
        'r2_validate': explained_variance_score(y_val.property_value, y_val.glm_pred)}, ignore_index= True)

    return metric_df
