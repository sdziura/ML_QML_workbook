from sklearn.datasets import load_iris                  # import dataset
from sklearn.preprocessing import StandardScaler        # For standarization of data
import numpy as np
import torch
import torch.nn as nn
from torcheval.metrics.functional import multiclass_f1_score
import random
from sklearn.model_selection import StratifiedKFold
from model import Two_Layer_Model

# Setting random seed
seed = 50
random.seed(seed)
torch.manual_seed(seed)

### HYPER-PARAMETERS ##############################################################################################

folds = 5
epochs = 5000

### PRE-PROCESSING ##############################################################################################


# Import Iris dataset as a Bunch, which is a dictionary-like object
# as_frame=True -> load data as a pd DataFrame
# returns data, target, feature_list, target_list, frame (data+target dataframe)
iris = load_iris(as_frame=True)

# Getting data and target Dataframes from the "iris" Bunch
X = iris.data
y = iris.target

# Standarizaiton of features to have a mean of 0 and std_dev of 1 in each feature using : 
# new_value = (old_value - feature_mean) / feature_std_dev
# feature_std_dev = sqrt( (sum(xi - mean)^2) / N )
# Reason: to treat all features equally
scaler = StandardScaler()
X = scaler.fit_transform(X)

        
### TRENING ##############################################################################################


# Podział danych na kilka (liczba zadana w zmiennej "folds") grup w celu przeprowadzenia walidacji krzyżowej (Cross Validation)
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

# Tablice do zapisu wyników
trainingEpoch_loss = np.empty((folds, epochs)) 
validationEpoch_loss = np.empty((folds, epochs))
y_test_list = []
y_test_pred_list = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    
    print(f"FOLD: {i}")
    X_train = torch.tensor(X[train_index], dtype=torch.float32)
    X_test = torch.tensor(X[test_index], dtype=torch.float32)
    y_train = torch.tensor(y[train_index].to_numpy(), dtype=torch.long)
    y_test = torch.tensor(y[test_index].to_numpy(), dtype=torch.long)
    
    model = Two_Layer_Model()

    # CrossEntropyLoss używa funkcje aktywacyjną Softmax.
    # Jej wzór to: S = e^x / sum(e^x)
    # Zwraca wartości 0-1. 
    # Suma wyjsciowych wartości będzie równa 1, więc można to traktować jako prawdopodobieństwo.
    # Gradienty tworzą Jakobian, z powodu użycia we wzorze wyjścia każdego neuronu. 
    # Na diagonali mamy: Si(1 - Si)
    # Na pozostałych: -Si*Sj

    # Następnie oblicza Loss przez:
    # -mean(actual * log(predicted))
    # Gdzie actual będzie 1 dla dobrego labela 
    loss_func = nn.CrossEntropyLoss()

    # Optymalizator Stochastic Gradent Descent (SGD), gdy podamy tylko parametry i learning rate, działa jak zwyczajny gradient descent, bez momentum.
    # Wartości parametrów zmienione są o gradient ((dL/dwi) pomnożony przez learning rate (weight = weight - learning_rate * gradient).
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(epochs):
        # Forward pass: Model oblicza predykcje dla wszystkich rekordów ze zbiory treningowego
        # Macierz cech x, jest mnożony przez warstwe liniową w postaci macierzy A: y = xA^T + b
        # Dla input_layer jest to dim(x)=[120,4], dim(A)=[32,4], dim(b)=[120,32] i dim(y)=[120,32]
        # Następnie macierz wynikowa y, przechodzi przez funkcje aktywacji ReLU: każda ujemna wartość jest zerowana.
        # Proces się powtarza przez kolejne warstwy. W ostatniej nie są zerowane ujemne wartości.
        y_pred = model(X_train)
        # Obliczanie wartości loss według wzorów -> -mean(actual * log(e^x / sum(e^x)))
        loss = loss_func(y_pred, y_train)
        # Kiedy troch tensor ma ustawione "requires_grad = True", to ten tensor zapisuje dodatkowe informacje
        # grad_fn -> zapisuje gradient działania, z którego został on wyprodukowany
        # grad -> wyznaczana wartość gradientu dla funkcji loss po danym tensorze, obliczana podczas backward pass
        loss.backward()
        trainingEpoch_loss[i][epoch] = loss.item()
        y_test_pred = model(X_test)
        validationEpoch_loss[i][epoch] = loss_func(y_test_pred, y_test).item()
        # Optymalizator zmienia wartości w tensroach na podstawie wyznaczonych gradientów.
        # Uzywa do tego wyznaczonej funkcji optymalizacyjnej (w tym przypadku prosty gradient descent: weight = weight - learning_rate * gradient))
        optimizer.step()
        # Po każdej iteracji trzeba wyzerować gradient, ponieważ normalnie akumulowałby gradient przy każdym uzyciu
        optimizer.zero_grad()

    
    # F1_score = 2*Precision*Recall/Precision+Recall
    # Gdzie: Precision = TP/(TP+FP)   Recall = TP/(TP+FN)
    score_test = multiclass_f1_score(y_test_pred, y_test, num_classes=3)
     
    # Konwesja logitów na przewidywany numer klasy
    y_test_pred_labels = torch.argmax(y_test_pred, dim=1)

    # Zapisywanie predykcji i prawdziwych podpisów
    y_test_list.append(y_test.numpy()) 
    y_test_pred_list.append(y_test_pred_labels.numpy())  

# Zapisanie wyników do pliku
np.savez("NN/results.npz", 
         train_loss=trainingEpoch_loss, 
         val_loss=validationEpoch_loss,
         y_test=y_test_list, 
         y_test_pred=y_test_pred_list)
print("Loss values saved to loss_values.npz")
