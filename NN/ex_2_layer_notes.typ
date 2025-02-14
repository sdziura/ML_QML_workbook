#set page(width: auto, height:auto)

= PRE-PROCESSING 


# Import Iris dataset as a Bunch, which is a dictionary-like object
# as_frame=True -> load data as a pd DataFrame
# returns data, target, feature_list, target_list, frame (data+target dataframe)
iris = load_iris(as_frame=True)

# Getting data and target Dataframes from the "iris" Bunch
X = iris.data
y = iris.target

# Standarizaiton of features to have a mean of 0 and std_dev of 1 in each feature using : 
# new_value = (old_value - feature_mean) / feature_std_dev
# feature_std_dev = sqrt( sum( (xi - mean)/N ))
# Reason: to treat all features equally
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Converting data type to torch.tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)


### MODEL ##############################################################################################

class Two_Layer_Model(nn.Module):
    def __init__(self):
        super().__init__() # Inicjalizacja dziedziczonego nn.Module
        # Rozmiar wejściowy to 4, ponieważ mamy 4 kolumny cech. Startowe wartości wag i bias to [-sqrt(k), sqrt(k)] dla k = 1/in_features
        # Ilość wag na każdą warstwe to in_features*out_features (np. dla pierwszej warstwy 4*32), oraz 1 bias na każdy neuron wyjściowy
        self.input_layer = nn.Linear(4, 32) 
        self.hidden_layer_1 = nn.Linear(32, 64)
        self.hidden_layer_2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 3) # Rozmiar wyjściowy to 3, ponieważ mamy 3 klasy 
        self.relu = nn.ReLU()

    def forward(self, x):
        # Funkcja aktywacyjna ReLU (Rectified Linear Unit): max(0,x)
        # Wyłącza niektóre neurony
        # Gradient równy 0 poniżej wartości 0 i 1 dla wartości powyżej 0
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer_1(out))
        out = self.relu(self.hidden_layer_2(out))
        # Output bez aktywcji, bo jako loss function Softmax Cross Entropy, które używa softmax
        out = self.output_layer(out)
        return out

        
### TRENING ##############################################################################################

model = Two_Layer_Model()

# CrossEntropyLoss używa funkcje aktywacyjną Softmax.
# Jej wzór to: e^x / sum(e^x)
# Zwraca wartości 0-1. 
# Suma wyjsciowych wartości będzie równa 1, więc można to traktować jako prawdopodobieństwo.
# Gradient tworzy Jakobian, z powodu użycia we wzorze wyjścia każdego neurona. 
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
for epoch in range(1000):
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
    # Optymalizator zmienia wartości w tensroach na podstawie wyznaczonych gradientów.
    # Uzywa do tego wyznaczonej funkcji optymalizacyjnej (w tym przypadku prosty gradient descent: weight = weight - learning_rate * gradient))
    optimizer.step()
    # Po każdej iteracji trzeba wyzerować gradient, ponieważ normalnie akumulowałby gradient przy każdym uzyciu
    optimizer.zero_grad()
    
    if epoch % 100 == 0:
        print('epoch ', epoch+1, ' loss = ', loss)


y_test_pred = model(X_test)
loss_test = loss_func(y_test_pred, y_test)
# F1_score = 2*Precision*Recall/Precision+Recall
# Gdzie: Precision = TP/(TP+FP)   Recall = TP/(TP+FN)
score_test = multiclass_f1_score(y_test_pred, y_test, num_classes=3)
print(f'\n TEST LOSS: {loss_test},  TEST SCORE: {score_test}')
print('\npred ', y_pred)
