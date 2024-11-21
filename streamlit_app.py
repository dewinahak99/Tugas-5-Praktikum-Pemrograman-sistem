import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Fungsi utama
def main():
    st.title("Aplikasi Klasifikasi Machine Learning")
    st.sidebar.title("Pengaturan Model")

    # Pilih model
    st.sidebar.write("Ivan Nahak")
    model_name = st.sidebar.selectbox("Model Klasifikasi", ["SVM", "Random Forest"])

    # Load dataset
    st.write("### Pilih Dataset")
    dataset_name = st.selectbox("Pilih dataset", ["Iris", "Upload dataset Anda"])
    if dataset_name == "Iris":
        data = load_iris(as_frame=True)
        X = data['data']
        y = data['target']
        st.write("Dataset Iris:")
        st.dataframe(X.head())
    else:
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Anda:")
            st.dataframe(df.head())
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

    # Split dataset
    test_size = st.sidebar.slider("Persentase data uji (%)", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if model_name == "SVM":
        # Parameter SVM
        st.sidebar.write("### Parameter Model SVM")
        kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)

        # Train SVM
        if st.sidebar.button("Latih Model"):
            model = SVC(kernel=kernel, C=C, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluasi Model
            tampilkan_hasil(model_name, y_test, y_pred, X, X_train, y_train, X_test)

    elif model_name == "Random Forest":
        # Parameter Random Forest
        st.sidebar.write("### Parameter Model Random Forest")
        n_estimators = st.sidebar.slider("Jumlah Estimators", 10, 200, 100, 10)
        max_depth = st.sidebar.slider("Kedalaman Maksimum", 1, 50, 10)

        # Train Random Forest
        if st.sidebar.button("Latih Model"):
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluasi Model
            tampilkan_hasil(model_name, y_test, y_pred, X, X_train, y_train, X_test)

# Fungsi untuk menampilkan hasil
def tampilkan_hasil(model_name, y_test, y_pred, X, X_train, y_train, X_test):
    st.write(f"### Hasil Klasifikasi ({model_name})")
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Akurasi:** {accuracy * 100:.2f}%")
    st.write("### Laporan Klasifikasi")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)


    # Visualisasi Data (jika ada 2 fitur)
    st.write("### Visualisasi Data (Hanya untuk 2 Fitur)")
    if X.shape[1] > 2:
        st.warning("Visualisasi hanya untuk dataset dengan 2 fitur. Dataset Anda memiliki lebih dari 2 fitur.")
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='viridis', label="Data Latih")
        plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='coolwarm', marker='x', label="Prediksi")
        plt.legend()
        plt.xlabel("Fitur 1")
        plt.ylabel("Fitur 2")
        st.pyplot(plt)
        

if __name__ == "__main__":
    main()