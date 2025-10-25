from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, RobustScaler)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, learning_curve
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Model:
    def __init__(self, model=None):
        if model is None:
            ## DIFFERENT MODELS:

            # self.model = LogisticRegression(class_weight="balanced", max_iter=2000)
            
            # self.model = LogisticRegression(
            #     solver="saga",
            #     max_iter=2000,
            #     C=10,
            #     penalty="l1",
            #     class_weight="balanced"
            # )

            self.model = GradientBoostingClassifier(
                n_estimators=500, 
                learning_rate=0.01, 
                max_depth=3,      
                min_samples_split=50, 
                min_samples_leaf=50, 
                subsample=0.7, 
                max_features=0.7,
                random_state=42
            )

            # self.model = RandomForestClassifier(
            #     n_estimators=200,
            #     max_depth=5,
            #     min_samples_split=20,
            #     random_state=42
            # )

            # self.model = XGBClassifier(
            #     n_estimators=800,
            #     learning_rate=0.01,
            #     max_depth=4,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     gamma=0.1,
            #     reg_lambda=1,
            #     eval_metric="logloss",
            #     random_state=42
            # )            
        else:
            self.model = model
        self.scaler = RobustScaler()
        self.sharpe = 0
        self.cross_value_accuracy = 0
        self.tscv = TimeSeriesSplit(n_splits=5)
    
    # Graphs the training vs test accuracy
    # Useful for debugging and determining whether the model is overfitting or underfitting
    # def plot_learning_curve(self, model, inputs, target, cv):
    #     train_sizes, train_scores, val_scores = learning_curve(
    #         model, inputs, target,
    #         cv=cv,
    #         train_sizes=np.linspace(0.1, 1.0, 10),
    #         scoring='accuracy',
    #         shuffle=False
    #     )

    #     # Mean + std for shading
    #     train_mean = np.mean(train_scores, axis=1)
    #     train_std = np.std(train_scores, axis=1)
    #     val_mean = np.mean(val_scores, axis=1)
    #     val_std = np.std(val_scores, axis=1)

    #     # Final validation accuracy (at largest training size)
    #     self.val_acc = val_mean[-1]
    #     print(f"Final Validation Accuracy: {self.val_acc:.4f}")

    #     # (Optional) also show final training accuracy
    #     final_train_acc = train_mean[-1]
    #     print(f"Final Training Accuracy: {final_train_acc:.4f}")        

    #     plt.figure(figsize=(8, 6))
    #     plt.plot(train_sizes, train_mean, 'o-', label="Training Accuracy")
    #     plt.plot(train_sizes, val_mean, 'o-', label="Validation Accuracy")
    #     plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
    #     plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2)
    #     plt.xlabel("Training Set Size")
    #     plt.ylabel("Accuracy")
    #     plt.title("Learning Curve")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    def train_model(self, inputs, target, daily_returns=None):
        inputs_train, inputs_test, target_train, target_test = train_test_split(inputs, target, test_size = 0.2, shuffle=False)

        inputs_train = self.scaler.fit_transform(inputs_train)
        inputs_test = self.scaler.transform(inputs_test)

        # Train the model
        self.model.fit(inputs_train, target_train)

        # Prints out how much of each indicator the model used when making predictions
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            for name, score in zip(inputs.columns, importances):
                print(f"{name}: {score:.4f}")        

        # Predictions
        target_predictions_test = self.model.predict(inputs_test)

        # Plots the learning curve
        # Useful for checking if model is overfitting or underfitting
        # self.plot_learning_curve(self.model, inputs, target, self.tscv)

        # Calculates the accuracy
        scores = cross_val_score(self.model, inputs.values, target.values, cv=self.tscv, scoring="accuracy")       
        self.cross_value_accuracy = scores.mean()     

        # Stats:
        # Prints the cross-value accuracy, Confusion matrix, and classification report.
        print("Classification Report:\n", classification_report(target_test, target_predictions_test))
        print("Confusion Matrix:\n", confusion_matrix(target_test, target_predictions_test))     
        print("Cross Value Accuracy: ", self.cross_value_accuracy)  
   

        if daily_returns is not None:
            probs = self.model.predict_proba(inputs_test)[:, 1]

            # Only try and trade when confident
            threshold = 0.55
            
            preds = np.where(probs > threshold, 1, 
                    np.where(probs < 1 - threshold, -1, 0))

            # Shift the positions to avoid lookahead bias
            positions = pd.Series(preds, index=target_test.index).shift(1)

            aligned_returns = daily_returns.loc[target_test.index]
            if isinstance(aligned_returns, pd.DataFrame):
                aligned_returns = aligned_returns.iloc[:, 0]

            strategy_returns = positions * aligned_returns

            # Compute the Sharpe ratio only if there are trades
            if strategy_returns.std() != 0:
                self.sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            else:
                self.sharpe = 0

            print("Sharpe Ratio: ", self.sharpe)
        
        return self.model
