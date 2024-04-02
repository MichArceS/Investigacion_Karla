import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df = pd.read_csv("correlacion_eficacia_creativa.csv")
corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()
