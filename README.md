# Zillow_Project
------
# Project Description 

This project will based off the Zillow dataset. We will be using the acquired Regression skills to attempt to predict house prices.
Zillow has many key data entries that can help in our prefiction models. Some of these include, but not limited to,  the total square feet, property value, tax value, bedroom count, and bathroom count. This predictions will be based off single family properties.

------
# Project Goals
+ Find the best predictors that are greatly linked to property value (taxvaluedollarcnt) 
+ Take the found drivers and develop a machine learning that can help predict house value
+ Make code reproducable

-----
# Data Dictionary


| Feature | Definition | Data Type |
| ----- | ----- | ----- |
| year_built | Year the house was built| `int` |
| property_value| Value of the property | `int` |
| tax_amount | Tax amount paid throughout the year| `int` |
| bedrooms| Total number of bedrooms the house has| `int`|
| bathrooms| Total number of bathrooms the house has| `float` |
| sqr_feet | Total square feet of the house | `float` |
| fips |   | `int` |


------
# Initial Hypothesis

My initial hypothesis is that bedroom count will be a huge preictor to property value (taxvaluedollarcnt)

------
# My Plan of Action

+ Aquire data from Codeup
  - Use Sequel Ace to obtain, and filter for the needed data that we will use for the project.
  
+ Prepare the acquired data
  - Download and turn Zillow data into a .csv
  - Bring in data into our jupyter notebook
  - Use pandas library
    - to clean data 
    - turn any columns from that need to be `object` to `int64`
    - turn any columns from that need to be `int64` to `object`
    - make `object` columns into numerical columns (No's and Yes's to  0's and 1's respectively)
  - Split our data into three parts 
    - `Train` which will have the largest amount of the data so we can use it to make our model
    - `Validate` which is used to make sure our model is the best it can before finally moving on
    - `Test` which is the final determinator to see if the model is good enough, the last part of modeling

Exploration
  
 
Modeling

- Use or make functions to help build our models 

- Build models 
  - Create our object
  - fit the model to find the most useful model
  - Use the best model on the in-sample data
  - Use the best model on the out-of-sample data
  
Delivery
  
  - Visualze to help the audiance easily see the findings of the work that was done
  - Model to help predict home value from Zillow
  
  ----
  # How to Reproduce the work
  
  - Clone this repo
  - Put data in the same file as the repo
  - Run the notebook
 

























