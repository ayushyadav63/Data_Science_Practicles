# Import necessary libraries
import seaborn as sns
import folium
from branca.element import Figure
import pandas as pd
from pywaffle.waffle import Waffle
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set seaborn color palette
current_palette = sns.color_palette
sns.set_palette(['blue'])

# Create a barplot using seaborn
sns.barplot(x=[1, 2, 3], y=[4, 5, 6])
plt.show()

# Install folium library
!pip install folium

# Create a folium map
m = folium.Map(location=[28.644800, 77.216721], zoom_start=11, min_zoom=8, max_zoom=14)
fig = Figure(width=700, height=500)
fig.add_child(m)

# Add different tile layers to the map
folium.TileLayer('OpenStreetMap').add_to(m)
folium.TileLayer('Stamen Terrain').add_to(m)
folium.TileLayer('Stamen Toner').add_to(m)
folium.TileLayer('Stamen Watercolor').add_to(m)
folium.TileLayer('CartoDB Positron').add_to(m)
folium.LayerControl().add_to(m)
m

# Create a sample DataFrame
ab = pd.DataFrame()
print(ab)

# Create another DataFrame with sample data
a = {'items': ['banana', 'apple', 'oats', 'milk', 'egg', 'chicken', 'meal'],
     'calories': [250, 300, 700, 66, 546, 636, 755]}
df = pd.DataFrame(a)
df

# Load a CSV file into a DataFrame
# df = pd.read_csv("https://raw.githubusercontent.com/codeforamerica/ohana-api/master/data/sample-csv/addresses.csv")

# Create a Waffle chart using pywaffle
a = [5, 4, 3, 2, 1]
plt.figure(
    FigureClass=Waffle,
    rows=5,
    columns=5,
    values=a,
    title={'label': 'the waffle chart', 'loc': 'right', 'size': 15},
    icons='face-smile'
)
plt.show()

# Load the Boston dataset and perform linear regression
boston = datasets.load_boston()
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
print("coefficient:", reg.coef)
print("Variance score:", format(reg.score(x_test, y_test)))

# Plot the residual errors
plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train, color="green", s=10, label='train data')
plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test, color="blue", s=10, label="test data")
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.legend(loc='upper right')
plt.title('Residual errors')
plt.show()

# Principal Component Analysis (PCA) for Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
target_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['target'] = df['target'].map(target_mapping)
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = StandardScaler().fit_transform(df[features])
pca = PCA(n_components=2)
principal_df = pd.DataFrame(data=pca.fit_transform(x), columns=['pc1', 'pc2'])
plt.figure(figsize=(8, 6))
scatter = plt.scatter(principal_df['pc1'], principal_df['pc2'], c=df['target'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2-D PCA')
legend1 = plt.colorbar(scatter, ticks=[0, 1, 2])
legend1.set_ticklabels(list(target_mapping.keys()))
plt.show()

# Linear Regression example using random data
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.rand(100, 1)
model = LinearRegression()
model.fit(x, y)
x_new = np.array([[0], [2]])
y_pred = model.predict(x_new)
plt.scatter(x, y)
plt.plot(x_new, y_pred, "r-")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Attractive Regression Plot
plt.scatter(x, y, color='red', marker='+')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.plot(x, model.predict(x), color='blue')
plt.title('Employee Details')
plt.show()

# Additional Code (Data Preprocessing, Visualization, Modeling) 

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy import stats

# Load your dataset
data = pd.read_csv('your_data.csv')

# Data Cleaning and Handling Missing Values
data.drop_duplicates(inplace=True)
data.dropna(subset=['column_name'], inplace=True)  # Remove rows with missing values in a specific column

# Data Visualization
# Example: Create a histogram and a box plot for a numerical feature
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data['numeric_feature'], bins=20, kde=True)
plt.title('Histogram')
plt.subplot(1, 2, 2)
sns.boxplot(y=data['numeric_feature'])
plt.title('Box Plot')
plt.show()

# Descriptive Statistics
summary_stats = data.describe()

# Feature Engineering (Example: Creating a new feature 'total_sales')
data['total_sales'] = data['quantity'] * data['price']

# Data Transformation (Example: Scaling numerical features)
scaler = StandardScaler()
data[['numeric_feature1', 'numeric_feature2']] = scaler.fit_transform(data[['numeric_feature1', 'numeric_feature2']])

# Bivariate Analysis (Example: Scatter plot for two numerical features)
sns.scatterplot(x='feature1', y='feature2', data=data)

# Outlier Detection (Example: Using Z-score for outlier detection)
z_scores = np.abs(stats.zscore(data['numeric_feature']))
data = data[(z_scores < 3)]  # Keep data points within 3 standard deviations

# Data Imputation (Example: Mean imputation for missing values)
data['column_name'].fillna(data['column_name'].mean(), inplace=True)

# Handling Categorical Data (Example: One-hot encoding)
data = pd.get_dummies(data, columns=['categorical_feature'])

# Data Splitting (Example: 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Data Scaling (Example: Min-Max scaling)
scaler = MinMaxScaler()
data[['numeric_feature1', 'numeric_feature2']] = scaler.fit_transform(data[['numeric_feature1', 'numeric_feature2']])

# EDA Documentation (Add comments and explanations to your code for documentation)

# Dimensionality Reduction (Example: PCA)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data[['numeric_feature1', 'numeric_feature2']])

# EDA Documentation (Summarize your findings and insights in comments)

# Now you can proceed to build and evaluate your machine learning models with the preprocessed data.

# Regression Example
# Generate random data for regression
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.rand(100, 1)

# Create and fit a Linear Regression model
model = LinearRegression()
model.fit(x, y)

# Predict for new data points
x_new = np.array([[0], [2]])
y_pred = model.predict(x_new)

# Plot the original data and the regression line
plt.scatter(x, y)
plt.plot(x_new, y_pred, "r-")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Example")
plt.show()

