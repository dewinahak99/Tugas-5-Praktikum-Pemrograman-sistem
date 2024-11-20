import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.exceptions import NotFittedError

st.title("Binary/Multiclass Classification Web App")
st.sidebar.title("Binary/Multiclass Classification Web App")
st.markdown("Upload your dataset and classify it!📊")
st.sidebar.markdown("Ivan Nahak")

# Fungsi untuk memuat data
def load_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Fungsi untuk memisahkan fitur dan target
def split(df, target_column):
    y = df[target_column]
    x = df.drop(columns=[target_column])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test

# Fungsi untuk menampilkan metrik evaluasi
def plot_metrics(metrics_list, model, x_test, y_test):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        if len(set(y_test)) == 2:  # ROC Curve hanya untuk binary classification
            RocCurveDisplay.from_estimator(model, x_test, y_test)
        else:  # Multiclass ROC Curve
            from sklearn.multiclass import OneVsRestClassifier
            from sklearn.metrics import roc_auc_score
            # Create OneVsRestClassifier for multiclass ROC
            model = OneVsRestClassifier(model)
            RocCurveDisplay.from_estimator(model, x_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
        st.pyplot()

# Widget untuk unggah file dataset
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Preview Dataset:")
        st.write(df.head())

        # Pilih kolom target
        target_column = st.sidebar.selectbox("Select target column", options=df.columns)
        
        # Encode kolom target jika diperlukan
        label_encoder = LabelEncoder()
        df[target_column] = label_encoder.fit_transform(df[target_column])

        # Split data
        x_train, x_test, y_train, y_test = split(df, target_column)

        # Pilih model
        st.sidebar.subheader("Choose Classifier")
        classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

        # Support Vector Machine (SVM)
        if classifier == 'Support Vector Machine (SVM)':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
            kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
            gamma = st.sidebar.radio("Gamma (Kernel Co efficient)", ("scale", "auto"), key='gamma')
            metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Support Vector Machine (SVM) Results")
                model = SVC(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                # Mengatur average berdasarkan apakah binary atau multiclass
                average_type = 'binary' if len(set(y_test)) == 2 else 'macro'
                st.write("Accuracy: ", accuracy_score(y_test, y_pred))
                st.write("Precision: ", precision_score(y_test, y_pred, average=average_type))
                st.write("Recall: ", recall_score(y_test, y_pred, average=average_type))
                plot_metrics(metrics, model, x_test, y_test)

        # Logistic Regression
        if classifier == 'Logistic Regression':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
            max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
            metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Logistic Regression Results")
                model = LogisticRegression(C=C, max_iter=max_iter)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                # Mengatur average berdasarkan apakah binary atau multiclass
                average_type = 'binary' if len(set(y_test)) == 2 else 'macro'
                st.write("Accuracy: ", accuracy_score(y_test, y_pred))
                st.write("Precision: ", precision_score(y_test, y_pred, average=average_type))
                st.write("Recall: ", recall_score(y_test, y_pred, average=average_type))
                plot_metrics(metrics, model, x_test, y_test)

        # Random Forest
        if classifier == 'Random Forest':
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators = st.sidebar.number_input("Number of trees", 100, 5000, step=10, key='n_estimators')
            max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1, key='max_depth')
            bootstrap = st.sidebar.radio("Bootstrap samples", [True, False], key='bootstrap')
            metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Random Forest Results")
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                # Mengatur average berdasarkan apakah binary atau multiclass
                average_type = 'binary' if len(set(y_test)) == 2 else 'macro'
                st.write("Accuracy: ", accuracy_score(y_test, y_pred))
                st.write("Precision: ", precision_score(y_test, y_pred, average=average_type))
                st.write("Recall: ", recall_score(y_test, y_pred, average=average_type))
                plot_metrics(metrics, model, x_test, y_test)

        # Show raw data option
        if st.sidebar.checkbox("Show raw data", False):
            st.subheader("Dataset")
            st.write(df)

else:
    st.write("Please upload a dataset to start.")