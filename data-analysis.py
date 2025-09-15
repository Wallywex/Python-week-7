# TASK ONE
# Importing the iris dataset in form of numpy arrays
from sklearn.datasets import load_iris
import pandas as pd

# Loading the dataset and converting to a Dataframe
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Adding the specie column
df['species_id'] = iris.target
# Making sure the name appears
df['species'] = df['species_id'].apply(lambda i :iris.target_names[i])

# converting from numpy array form to CSV 
df.to_csv("iris_dataset.csv", index= False)

# Loading from CSV file we created
df2 = pd.read_csv("iris_dataset.csv")
print("Displaying the first few rows")
print(df2.head(), '\n')

# Exploring the dataset
# Checking shape.(Rows x Columns)
print("The shape of the dataset is")
print(df2.shape, '\n')

# View column names
print("The columns in the dataset are")
print(df2.columns, '\n')

# Checking for data types and any missing values
print("The info of the dataset, including data types and missing values is")
print(df2.info(), '\n')

# Findings and observations
# There are no missing values. The iris dataset is clean.
# Each species (setosa, versicolor, virginica) has 50 samples each making it a perfectly balanced dataset.


# TASK TWO

# Using the describe() method to compute statistics
print("The statistical summary of the dataset is")
print(df.describe(), '\n')

# Grouping Categorical Columns and computing the mean accordingly

# Grouping by species, the mean for each column is
grouped_species = df2.groupby('species').mean()
print("Grouping by species, the mean for each column is")
print(grouped_species, '\n')

# Grouping by species_id, the mean for each column is
grouped_species_id = df2.groupby('species_id').mean(numeric_only=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
print("Grouping by species_id, the mean for each column is")
print(grouped_species_id)


"""An interesting finding I discovered in this analysis
is that Pandas, by default, will limit the number of 
columns you see. This took a long time debugging.
Additionally,the average dimension of the setosa is 
smaller compared to the versicolor, and the average 
dimension of the versicolor is smaller compared to 
the versilica."""

# Task 3 (DATA VISUALIZATION)

# To visualize the data, we will use matplotlib and seaborn libraries
# check data-visualization.py




