import pandas as pd
import numpy as np
import os
from env import get_connection
import matplotlib.pyplot as plt
import seaborn as sns


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
        
        
