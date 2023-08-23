#libraries
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
#%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
plt.style.use('ggplot') #Applying style to graphs
import termcolor #for customization of text

#for univariate plots
def plot_percentage(df, x):
    sns.set(style="darkgrid")

    # create the bar chart
    ax = sns.barplot(y=df.index, x="Percentage", orient = 'h', data=df)

    # set the chart title and axis labels
    plt.title("Percentage distribution of " + x)
    plt.ylabel(x)
    plt.xlabel("Percentage")

    # add percentage values on top of each bar
    for i in ax.containers:
        ax.bar_label(i, label_type='edge', fontsize=8, rotation='horizontal')

    # rotate the x-axis labels
    #plt.xticks(rotation='horizontal')

    # show the chart
    plt.show()
#count percentages

def count_percentage(df, column):
    counts = df[column].value_counts()
    percentages = counts / counts.sum() * 100
    _df = pd.concat([counts, percentages], axis=1)
    _df.columns = ['Count', 'Percentage']
    _df = _df.sort_values(by='Percentage', ascending=False)
    _df = _df[_df['Count'] >= 0]
    return _df