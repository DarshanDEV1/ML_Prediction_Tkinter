import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.preprocessing import StandardScaler

class MLModelApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Model Prediction App")
        self.geometry("500x400")
        self.dataset = None
        self.target_column = None
        self.algorithm = tk.StringVar(value="Random Forest")
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Select a dataset:").pack(pady=5)
        tk.Button(self, text="Load Dataset", command=self.load_dataset).pack(pady=5)

        tk.Label(self, text="Select the target column:").pack(pady=5)
        self.target_column_entry = tk.Entry(self)
        self.target_column_entry.pack(pady=5)

        tk.Label(self, text="Choose an algorithm:").pack(pady=5)
        tk.Radiobutton(self, text="Random Forest", variable=self.algorithm, value="Random Forest").pack(anchor=tk.W)
        tk.Radiobutton(self, text="KNN", variable=self.algorithm, value="KNN").pack(anchor=tk.W)

        tk.Button(self, text="Predict", command=self.predict_data).pack(pady=10)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.dataset = pd.read_csv(file_path)
            messagebox.showinfo("Dataset Loaded", "Dataset loaded successfully!")

    def predict_data(self):
        if self.dataset is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return

        self.target_column = self.target_column_entry.get()
        if not self.target_column:
            messagebox.showerror("Error", "Please enter the target column name.")
            return

        if self.target_column not in self.dataset.columns:
            messagebox.showerror("Error", "Target column not found in the dataset.")
            return

        # Preprocess the dataset to handle missing values and categorical features
        numeric_columns = self.dataset.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        self.dataset[numeric_columns] = imputer.fit_transform(self.dataset[numeric_columns])

        categorical_columns = self.dataset.select_dtypes(include=[object]).columns
        if len(categorical_columns) > 0:
            encoder = OneHotEncoder(sparse=False, drop='first')
            encoded_data = encoder.fit_transform(self.dataset[categorical_columns])
            self.dataset = self.dataset.drop(categorical_columns, axis=1)
            # encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names(categorical_columns))
            # self.dataset = pd.concat([self.dataset, encoded_df], axis=1)

        x = self.dataset.drop(self.target_column, axis=1)
        y = self.dataset[self.target_column]

        if not np.issubdtype(y.dtype, np.number):
            messagebox.showerror("Error", "Target column must be numeric.")
            return

        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Normalize the features using StandardScaler
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Try different values for n_neighbors
        n_neighbors = [3, 5, 7]  # You can try more values if needed

        for n in n_neighbors:
            model = KNeighborsRegressor(n_neighbors=n)
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)

            self.plot_graph(y_test, y_pred)
            plt.title(f"KNN (n_neighbors={n})")
            plt.show()

        if self.algorithm.get() == "Random Forest":
            model = RandomForestRegressor()
        elif self.algorithm.get() == "KNN":
            model = KNeighborsRegressor()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        self.plot_graph(y_test, y_pred)

        output_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if output_file_path:
            predicted_df = pd.DataFrame({self.target_column: y_test, "Predicted": y_pred})
            predicted_df.to_csv(output_file_path, index=False)
            messagebox.showinfo("Prediction Saved", "Predicted dataset saved successfully!")

    def plot_graph(self, y_test, y_pred):
        plt.scatter(y_test, y_pred)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("True Values vs. Predictions")
        plt.show()

if __name__ == "__main__":
    app = MLModelApp()
    app.mainloop()
