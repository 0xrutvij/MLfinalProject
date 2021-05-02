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
    
    feature_names = np.array(['contact_class_score_diff','contact_id','counter','delay','msg_delay','msg_delay_stamps'])
    df = pd.read_csv('./processedData/aggregatedAndProcessed.csv')
    x = df[['contact_class_score_diff','contact_id','counter','delay','msg_delay','msg_delay_stamps']]
    y = df['contact_type']
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