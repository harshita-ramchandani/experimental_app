import streamlit as st
from sklearn.datasets import make_classification
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import time 
import pickle
import os
from sklearn.pipeline import make_pipeline

st.set_page_config(
    page_title="Linear & Logistic",
    page_icon="👋",
)

st.title("Logistic Regression")
st.subheader("Forward and Backward Feature Selection")
st.header("Machine Learning")
st.markdown("---")

X, y = make_classification(n_samples=100, n_features=20, n_informative=10,
                            n_redundant=2, n_repeated=0, n_classes=2,
                            class_sep=2.0, random_state=42)

lr = LogisticRegression()
lr.fit(X, y)    

sfs = SequentialFeatureSelector(lr, direction='forward', n_features_to_select='auto')
sbs = SequentialFeatureSelector(lr, direction='backward', n_features_to_select='auto')

n_features_fwd = []
scores_fwd = []
n_features_bwd = []
scores_bwd = []

with st.spinner(text='Please Wait..'):
    for i in range(1,16):
        sfs.n_features_to_select = i
        sfs_pipeline = make_pipeline(sfs, lr)
        sfs_pipeline.fit(X, y)
        score_fwd = accuracy_score(y, sfs_pipeline.predict(X))
        n_features_fwd.append(i)
        scores_fwd.append(score_fwd)

        sbs.n_features_to_select = i
        sbs_pipeline = make_pipeline(sbs, lr)
        sbs_pipeline.fit(X, y)
        score_bwd = accuracy_score(y, sbs_pipeline.predict(X))
        n_features_bwd.append(i)
        scores_bwd.append(score_bwd)
        #st.write(score_bwd)

        st.write(f'Iteration {i}: Forward score :{score_fwd:.4f}, Backward score : {score_bwd:.4f}')

        #time.sleep(0.05)

    st.balloons()
    st.success('Done!')




@st.cache_data
def plot_scores(n):
    fig=plt.figure()
    plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")
    plt.plot(n_features_fwd[:n], scores_fwd[:n], label='Forward')
    plt.plot(n_features_bwd[:n], scores_bwd[:n], label='Backward')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Sequential Feature Selection')
    plt.legend()
    st.pyplot(fig)

if _name_ == '_main_':
    iterations_slider = st.slider(label="Iterations", min_value=1, max_value=len(n_features_fwd), value=len(n_features_fwd))
    interact(plot_scores, n=iterations_slider)
