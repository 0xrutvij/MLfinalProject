#from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics

pred = [.99,.98,.72,.70,.65,.51,.39,.24,.11,.01]
y = [1,1,0,1,1,0,0,1,0,0]


"""
score = roc_auc_score(y, pred)
fpr, tpr, _ = roc_curve(y, pred, drop_intermediate=False)

print(score, fpr, tpr)

plt.plot(fpr, tpr, marker='.', label='Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
"""

fpr, tpr, thresholds = metrics.roc_curve(y, pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,                                          estimator_name='example estimator')
display.plot()
plt.show()
