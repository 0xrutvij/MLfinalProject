from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
import pandas as pd
import seaborn as sns

def correlation_heatmap(train):
        correlations = train.corr(method='spearman')

        fig, ax = plt.subplots(figsize=(6,6))
        sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
                    )
        plt.show()

if __name__ == '__main__':
        
        df = pd.read_csv('./processedData/aggregatedAndProcessed.csv')
        df.rename(columns={
        "contact_type":"contactType",
        "contact_class_score_diff": "classScore", 
        "contact_id": "deviceContacted", 
        "msg_delay_stamps":"serverDelay", 
        "msg_delay":"deviceDelay"}, inplace=True)
        print(df.columns)

        feature_names = np.array(['classScore', 'deviceContacted', 'counter',
       'delay', 'deviceDelay', 'serverDelay'])
        x = df[['classScore', 'deviceContacted', 'counter','delay', 'deviceDelay', 'serverDelay']]
        y = df['contactType']
        df.drop('Unnamed: 0', axis=1, inplace=True)
        correlation_heatmap(df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=6375)
        xgb = XGBRegressor(n_estimators=100)
        xgb.fit(x_train, y_train)
        x_test = np.ascontiguousarray(x_test)
        y_test = np.ascontiguousarray(y_test)
        print(xgb.feature_importances_)

        perm_importance = permutation_importance(xgb, x_test, y_test)
        sorted_idx = perm_importance.importances_mean.argsort()
        plt.barh(feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx])
        plt.show()
        correlation_heatmap(x_train[feature_names[sorted_idx]])