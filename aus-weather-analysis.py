# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

# %%
df = pd.read_csv("weatherAUS.csv")

# TEMPORARY
# df.fillna(value=0, inplace=True)

df.Date = pd.to_datetime(df.Date)

print(df.shape, "\n")
print(df.count().sort_values(), "\n")
# Sunshine, Evaporation, Cloud3pm, Cloud9am all have > 54,000 missing values
# Drop them
features_to_drop = ['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am']
df.drop(features_to_drop, axis=1, inplace=True)

print(df.columns, "\n")

sb.set(font_scale=0.75)
sb.heatmap(df.corr(), annot=True)

# %%
for col, values in df.items():
    if values.dtype != 'object':
        print("plotting ", col)
        plt.hist(values, facecolor='peru', edgecolor='blue')
        plt.title(col)
        plt.show()

# %%

