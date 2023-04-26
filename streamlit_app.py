# import streamlit as st
# from sklearn.datasets import make_classification
# from sklearn.feature_selection import SequentialFeatureSelector
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# # set page title
# st.set_page_config(page_title="Sequential Feature Selection")

# # # generate the dataset outside the app function
# # X, y = make_classification(n_samples=100, n_features=20, n_informative=15,
# #                            n_redundant=2, n_repeated=0, n_classes=2,
# #                            class_sep=2.0, random_state=42)

# # # create a sequential forward feature selector
# # sfs = SequentialFeatureSelector(LogisticRegression(), direction='forward', n_features_to_select=10)

# # # create a sequential backward feature selector
# # sbs = SequentialFeatureSelector(LogisticRegression(), direction='backward', n_features_to_select=10)

# # # initialize empty lists to store the number of selected features and corresponding scores
# # n_features_fwd = []
# # scores_fwd = []
# # n_features_bwd = []
# # scores_bwd = []

# # # loop through a range of iterations to select a variable number of features
# # for i in range(1, 10):
# #     # create a new instance of the logistic regression model
# #     lr = LogisticRegression()
# #     sfs = SequentialFeatureSelector(LogisticRegression(), direction='forward', n_features_to_select=i) 
# #     # fit the sequential forward feature selector
# #     sfs.fit(X, y)
# #     # get the selected features and their corresponding scores
# #     selected_features_fwd = sfs.transform(X)
# #     lr.fit(selected_features_fwd, y)
# #     score_fwd = accuracy_score(y, lr.predict(selected_features_fwd))
# #     # append the number of selected features and the corresponding score to the lists
# #     n_features_fwd.append(i)
# #     scores_fwd.append(score_fwd)

# #     # create a new instance of the logistic regression model
# #     lr = LogisticRegression()
# #     sbs = SequentialFeatureSelector(LogisticRegression(), direction='backward', n_features_to_select=i)
# #     # fit the sequential backward feature selector
# #     sbs.fit(X, y)
# #     # get the selected features and their corresponding scores
# #     selected_features_bwd = sbs.transform(X)
# #     lr.fit(selected_features_bwd, y)
# #     score_bwd = accuracy_score(y, lr.predict(selected_features_bwd))
# #     # append the number of selected features and the corresponding score to the lists
# #     n_features_bwd.append(i)
# #     scores_bwd.append(score_bwd)

# # define the Streamlit app function
# def app():
# #     # define a function to plot the scores
# #     def plot_scores(n):
# #         fig, ax = plt.subplots()
# #         ax.plot(n_features_fwd[:n], scores_fwd[:n], label='Forward')
# #         ax.plot(n_features_bwd[:n], scores_bwd[:n], label='Backward')
# #         ax.set_xlabel('Number of Features')
# #         ax.set_ylabel('Accuracy')
# #         ax.set_title('Sequential Feature Selection')
# #         ax.legend()
# #         st.pyplot(fig)

# #     # create a slider for the number of iterations
# #     iterations_slider = st.slider(min_value=1, max_value=len(n_features_fwd), value=len(n_features_fwd), step=1, label='Iterations:')

# #     # display the plot
# #     plot_scores(iterations_slider)

#   st.write("hello world")

# # # call the app function
# if __name__ == "__main__":
#     app()




import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Set the page title
st.set_page_config(page_title="Logistic Regression Demo")
st.set_option('deprecation.showPyplotGlobalUse', False)


# Define the logistic regression function
def logistic_regression(X, y, epochs):
    # Initialize the logistic regression model
    print(epochs)
    model = LogisticRegression(solver='saga', max_iter=epochs)

    # Train the model
    model.fit(X, y)

    # Predict the class labels for the input data
    y_pred = model.predict(X)

    return model, y_pred, model.coef_, model.intercept_

# Define the function to plot the decision boundary
def plot_decision_boundary(X, y, model):
    # Create a meshgrid of points to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict the class labels for the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and the data points
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

# Define the Streamlit app
def app():
    # Set the app title
    st.title("Logistic Regression Demo")

    # Define the dataset selection dropdown
    dataset = st.selectbox(
        'Select a dataset:',
        ('Iris', 'Breast Cancer', 'Wine')
    )

    # Load the selected dataset
    if dataset == 'Iris':
        X, y = datasets.load_iris(return_X_y=True)
        X = X[:, :2]  # we only take the first two features.
    elif dataset == 'Breast Cancer':
        X, y = datasets.load_breast_cancer(return_X_y=True)
        X = X[:, :2]
    elif dataset == 'Wine':
        X, y = datasets.load_wine(return_X_y=True)
        X = X[:, :2]
    else:
        st.error("Invalid dataset selected.")

    # Define the epochs slider
    epochs = st.slider("Epochs", 0, 1000, 0, 5)

    # Plot the decision boundary for the selected epochs
    if epochs > 0:
        model, y_pred, coef, intercept = logistic_regression(X, y, epochs)
        plt.figure()
        plot_decision_boundary(X, y, model)
        st.pyplot()
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        st.pyplot()


# Run the app
if __name__ == "__main__":
    app()
