import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    df = pd.read_csv("iris_dataset.csv")
except FileNotFoundError:
    print('Error: CSV file not found')
    df = None
except pd.errors.EmptyDataError:
    print('Error: CSV file is empty')
    df = None
except pd.errors.ParserError:
    print('Error: CSV file is corrupted or invalid')
    df = None

# Line Plot to show the trend of sepal and petal lengths across all samples
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['sepal length (cm)'], label = 'sepal length')
plt.plot(df.index, df['petal length (cm)'], label = 'Petal length')

plt.xlabel('Sample Index')
plt.ylabel('Length (cm)')

plt.title('Trend of sepal and petal length across samples')
plt.grid(True)
plt.legend()



# Bar chart to show the average Petal length per species

avg_petal_length = df.groupby('species')['petal length (cm)'].mean()
plt.figure(figsize=(8,6))
plt.bar(avg_petal_length.index, avg_petal_length.values)
plt.xlabel('Species')
plt.ylabel('Average petal length')
plt.title('Average Petal length by specie')


# Histogram to show the distribution of the petal width

plt.figure(figsize=(8,6))
plt.hist(df['petal width (cm)'], bins=10)
plt.xlabel('Petal width')
plt.ylabel('Frequency')
plt.title('Distribution of petal widths across the dataset')
plt.grid(True, axis='y')


# Scatter plot using seaborn to show sepal vs petal length by species
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x = 'sepal length (cm)', y='petal length (cm)', 
hue='species', palette='Set1')
plt.title('Sepal vs Petal length')
plt.show()