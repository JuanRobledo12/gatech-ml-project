import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

class MultiplePlotMaker:

    def __init__(self, df):
        self.df = df

    def get_column_names(self):
        column_names = list(self.df.columns)
        return column_names
    
    def plot_multiple_distributions(self):
        column_names = self.get_column_names()
    
        # Creating plots to check distributions in the dataset
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        # Plot each column
        for i, col in enumerate(column_names):
            if col == 'Age':  # Use bar plot for discrete variables
                sns.countplot(x=self.df[col], ax=axes[i])
                axes[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure ticks are integers
                axes[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Adjust tick frequency
                axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels if needed
            elif col in ['Pregnancies', 'Outcome']:
                sns.countplot(x=self.df[col], ax=axes[i])
            else:  # Use histogram for continuous variables
                sns.histplot(self.df[col], kde=False, ax=axes[i])
            
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def plot_multiple_boxplots(self):
        column_names = self.get_column_names()
        column_names.pop(-1)  # Assuming the last column is not to be plotted

        # Create boxplots to identify outliers
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        # Plot each boxplot
        for i, col in enumerate(column_names):
            sns.boxplot(x=self.df[col], ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Value')

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()
    
    def plot_multiple_side_by_side_boxplots(self):
        column_names = self.get_column_names()
        target_variable_name = column_names.pop(-1)  # Assuming the last column is the target

        # Create boxplots to identify outliers
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        # Plot each boxplot
        for i, col in enumerate(column_names):
            sns.boxplot(data=self.df, x=target_variable_name, y=col, ax=axes[i])
            axes[i].set_title(f'Side-by-side boxplot of {col}')

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def plot_multiple_overlapping_hist(self):
        column_names = self.get_column_names()
        target_variable_name = column_names.pop(-1)  # Assuming the last column is the target

        # Create subplots
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        # Loop through each column to create overlapping histograms
        for i, col in enumerate(column_names):
            subset1 = self.df[self.df[target_variable_name] == 0][col]
            subset2 = self.df[self.df[target_variable_name] == 1][col]

            axes[i].hist(subset1, color="blue", label="0", density=True, alpha=0.5)
            axes[i].hist(subset2, color="red", label="1", density=True, alpha=0.5)

            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'Overlapping Histogram of {col}')
            axes[i].legend()

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()


class StatAnalysisTools:

    def __init__(self, df):
        self.df = df

    def perform_t_test(self):

        target_column_name = self.df.columns[-1] 
        # Assuming df is your DataFrame and 'Outcome' is the target variable
        for col in self.df.columns[:-1]:  # Exclude the target column
            group1 = self.df[self.df[target_column_name] == 0][col]
            group2 = self.df[self.df[target_column_name] == 1][col]
            t_stat, p_value = ttest_ind(group1, group2)
            print(f"{col}: t_stat = {t_stat},  p-value = {p_value}")
    
    def perform_correlation_binary_target(self):
        '''
        the point-biserial correlation (a special case of Pearson correlation for binary target variables) 
        to see how strongly each feature is associated with the target.
        '''

        for column in self.df.columns[:-1]:  # Exclude the target column
            correlation, p_value = pointbiserialr(self.df[column], self.df['Outcome'])
            print(f"{column}: correlation = {correlation}, p-value = {p_value}")

class ModelEvaluation:
    def __init__(self) -> None:
        pass
    
    def evaluate_classifier(self, X, y, trained_model):

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        # Assuming you have a separate test set
        y_pred = trained_model.predict(X_test)
        y_pred_proba = trained_model.predict_proba(X_test)[:, 1]  # For ROC-AUC if needed

        # Calculate various metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Print the results
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"ROC-AUC Score: {roc_auc:.3f}")
        print("Confusion Matrix:")
        print(conf_matrix)

        # Plot confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()
