import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# Import the function that will download the dataset.
from sklearn.datasets import fetch_california_housing

pd.set_option('display.max_columns', None)  # i can see all the colums

# Load the sklearn version of the California Housing dataset.
ds = fetch_california_housing()

keys = ds.keys()
print(keys)

print(ds['DESCR'])

# Get reasonable column names
column_names = ds['feature_names']
print(column_names)

df = pd.DataFrame(data=ds['data'], columns=column_names)

# Create a Pandas Series for the target values
targets = pd.Series(data=ds['target'], name='MedHouseValue')

print(targets)

# Display the first few rows of the DataFrame and the Series
print(df.head())
print(targets.head())

print()

print("Summary info")

print()

summary = df.describe()
print(summary)  # We can see that the row 'count' is the same everywhere

plt.hist(targets, bins=100, color='skyblue', edgecolor='black')

plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.title('Distribution of Target Values (Matplotlib Histogram)')

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))  # Adjust the figure size if needed
sns.histplot(targets, bins=100, color='skyblue', kde=True)

# Add labels and a title
plt.xlabel('Median House Value')
plt.ylabel('Density')
plt.title('Distribution of Target Values (Seaborn distplot)')

# Show the plot
plt.show()

plt.figure(figsize=(12, 12))
for (i, col) in enumerate(df.columns):
    plt.subplot(3, 3, i+1)
    plt.hist(df[col], 30)
    plt.title(df[col].name)

df.plot(subplots = True, figsize = (30, 30))

# Your code here.

# Split data into 75-25 train/test split -- replace the [None]*4 with your code.
(Xtr, Xte, ytr, yte) = train_test_split(df, targets, train_size=0.75)

(Xtr.shape, Xte.shape)


# Your code here.
model = LinearRegression()
model.fit(Xtr, ytr)

model.predict(Xte)

model.coef_, model.intercept_


preds_tr = model.predict(Xtr)
preds_te = model.predict(Xte)

print(f'RMSE on train: {np.sqrt(mean_absolute_error(preds_tr, ytr))}')
print(f'RMSE on train: {np.sqrt(mean_squared_error(preds_te, yte))}')
print(f'MAE on train: {mean_absolute_error(preds_tr, ytr)}')
print(f'MAE on train: {mean_squared_error(preds_te, yte)}')

np.sqrt(np.mean((model.predict(Xte) - yte)**2.0))

np.sqrt(np.mean((model.predict(Xtr) - ytr)**2.0))


"""from sklearn.linear_model import LinearRegression

pred = model.predict(Xtr)
residuals = ytr - pred

plt.scatter(pred, residuals)

plt.hist"""

(Xtr, Xte, ytr, yte) = train_test_split(df, targets, train_size=0.5)

model = LinearRegression()
model.fit(Xtr, ytr)

def resplot(y, preds):
    plt.scatter(preds, preds-y, s = 2)
resplot(model.predict(df), targets)
