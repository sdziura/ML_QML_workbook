import numpy as np
import matplotlib
from sklearn.metrics import f1_score

# Usuwa ostrzeżenie 
matplotlib.use('TkAgg')  

# Wczytaj zapisane dane z treningu
data = np.load("SVM/results.npz", allow_pickle=True)
y_test_list = data["y_test"]
y_customSVN_pred_list = data["y_SVN_preds"]
y_linearSVC_pred_list = data["y_linearSVC_preds"]
y_SVC_kernel_linear_pred_list = data["y_SVC_kernel_linear_preds"]

f1_custom = []
f1_linearSVC = []
f1_SVC = []
for fold in range(len(y_test_list)):
    f1_custom.append(f1_score(y_test_list[fold], y_customSVN_pred_list[fold], average='macro'))
    f1_linearSVC.append(f1_score(y_test_list[fold], y_linearSVC_pred_list[fold], average='macro'))
    f1_SVC.append(f1_score(y_test_list[fold], y_SVC_kernel_linear_pred_list[fold], average='macro'))

print(f"F1 SCORES (mean):")
print(f"\tCustom SVM:\t {np.mean(f1_custom):.4f}")
print(f"\tLinear SVC:\t {np.mean(f1_linearSVC):.4f}")
print(f"\tSVC- linear:\t {np.mean(f1_SVC):.4f}")