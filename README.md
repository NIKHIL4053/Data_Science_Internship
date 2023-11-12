# Titanic Survival Prediction using Python and Machine Learning

This project aims to predict the survival of passengers on the Titanic using Python and machine learning techniques. The project uses the Titanic dataset from Kaggle, which contains information about the passengers such as their name, age, gender, ticket class, fare, cabin, and whether they survived or not.

## Libraries used

The project uses the following Python libraries:

- Numpy: for numerical computations and array operations
- Matplotlib: for data visualization and plotting
- Seaborn: for statistical data visualization and styling
- Sklearn: for machine learning algorithms and tools

## Model used

The project uses the **Logistic Regression** model from sklearn, which is a supervised learning algorithm for binary classification. Logistic Regression predicts the probability of an outcome that can only have two values, such as 0 or 1, yes or no, etc. In this case, the outcome is whether a passenger survived the Titanic disaster or not.

## Data file

The project uses the `tested.csv` file as the input data, which contains 418 rows and 11 columns. The columns are:

- PassengerId: a unique identifier for each passenger
- Pclass: the ticket class of the passenger (1 = 1st, 2 = 2nd, 3 = 3rd)
- Name: the name of the passenger
- Sex: the gender of the passenger (male or female)
- Age: the age of the passenger in years
- SibSp: the number of siblings and spouses aboard the Titanic
- Parch: the number of parents and children aboard the Titanic
- Ticket: the ticket number of the passenger
- Fare: the passenger fare
- Cabin: the cabin number of the passenger
- Embarked: the port of embarkation of the passenger (C = Cherbourg, Q = Queenstown, S = Southampton)

## How to run the project

To run the project, you need to have Python 3 and the required libraries installed on your system. You can use the following commands to install the libraries:

```python
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

Then, you can clone this repository or download the files to your local machine. You can use the following command to clone the repository:

```git
git clone https://github.com/NIKHIL4053/Titanic_Survival_prediction.git
```

After that, you can open the `kernel.ipynb` file in a Jupyter Notebook or any other Python IDE and run the code cells. The code will load the data, perform some exploratory data analysis, preprocess the data, train the Logistic Regression model, and make predictions on the test data. The predictions will be saved in a file called `submission.csv`, which you can submit to Kaggle to see your score.

## Results and evaluation

The project achieves an accuracy score of 0.78186 on the test data, which ranks in the top 10% of the Kaggle leaderboard. The project also evaluates the model using various metrics such as confusion matrix, precision, recall, f1-score, and roc curve. The results show that the model performs well on both the training and test data, and has a good balance between sensitivity and specificity.
