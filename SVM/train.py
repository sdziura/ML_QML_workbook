import random
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from model import SVM
from sklearn.svm import SVC, LinearSVC

#warnings.filterwarnings('ignore', 'ConvergenceWarning*')

# Setting random seed
seed = 50
random.seed(seed)

### HYPER-PARAMETERS ##############################################################################################

folds = 5
lr=0.001
lambda_p=0.01
iters=5000

C=1/(2*lambda_p)

### PRE-PROCESSING ##############################################################################################

data = load_breast_cancer()
X = data.data
y = data.target
y = np.where(y == 0, -1, 1)

# Standarizaiton of features to have a mean of 0 and std_dev of 1 in each feature using : 
# new_value = (old_value - feature_mean) / feature_std_dev
scaler = StandardScaler()
X = scaler.fit_transform(X)

### TRENING ##############################################################################################


# Podział danych na kilka (liczba zadana w zmiennej "folds") grup w celu przeprowadzenia walidacji krzyżowej (Cross Validation)
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

# Tablice do zapisu wyników
y_test_list = []
y_customSVN_pred_list = []
y_linearSVC_pred_list = []
y_SVC_kernel_linear_pred_list = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    # Tworzenie nowych modeli na każdy etap walidacji krzyżowej
    custom_SVM = SVM(n_iters=iters, learning_rate=lr, lambda_param=lambda_p)
    # L2 penelty w LinearSVC jest tożsamy karą za złą klasyfikacje użytą w customowym SVM powyżej (tylko korzysta z hipermarapetru "C", zamiast "lambda") 
    # W przeciwieństwie do ręcznie zaimplementowanego SVM powyżej, 
    # LinearSVC korzysta z koordynowanego spadku (Coordinate Descent), zamiast stochastycznego spadku gradientu (SGD)
    linear_SVC = LinearSVC(C=C, loss="hinge", penalty="l2", max_iter=iters, random_state=seed)
    # SVC z biblioteki sklearn nie korzysta z learning_rate, 
    # ponieważ optymalizuje parametry przez zamiane na problem kwadratowy (Quadratic Problem), a nie metodą gradientu spadkowego (Gradient Descent)
    # SVC korzysta z hiperparamtru "C" zamiast "lambdy", ale działają na tej podobnej zasadzie.
    SVC_kernel_linear = SVC(kernel='linear', max_iter=iters, C=C, random_state=seed, degree=1)
    
    custom_SVM.fit(X[train_index], y[train_index])
    linear_SVC.fit(X[train_index], y[train_index])
    SVC_kernel_linear.fit(X[train_index], y[train_index])
    
    y_test_list.append(
        y[test_index]
        )
    y_customSVN_pred_list.append(
        custom_SVM.predict(X[test_index])
        )
    y_linearSVC_pred_list.append(
        linear_SVC.predict(X[test_index])
        )
    y_SVC_kernel_linear_pred_list.append(
        SVC_kernel_linear.predict(X[test_index])
        )

# Zapisanie wyników do pliku
np.savez("SVM/results.npz", 
         y_test=np.array(y_test_list, dtype=object), 
         y_SVN_preds=np.array(y_customSVN_pred_list, dtype=object),
         y_linearSVC_preds=np.array(y_linearSVC_pred_list, dtype=object),
         y_SVC_kernel_linear_preds=np.array(y_SVC_kernel_linear_pred_list, dtype=object))
print("Predictions saved to results.npz")
