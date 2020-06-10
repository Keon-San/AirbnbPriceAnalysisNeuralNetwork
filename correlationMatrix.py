import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

dataframe = pd.read_csv(r'AB_NYC_2019.csv')

corrMatrix = dataframe.corr()

sn.heatmap(corrMatrix, annot=True)
plt.show()
