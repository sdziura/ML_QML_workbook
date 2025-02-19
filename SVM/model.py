import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        # Parametr lambda parametryzuje wagę dla szerokości pasa wyznaczonego przez marginesy
        # Większa lambda -> SVM wyznaczy jak najszerszy pas, kosztem większej ilości błędnych klasyfikacji
        # Mneijsza lambda -> SVM wynzaczy węższy pas, skupiając się na precyzyjnym rozdzieleniu klas
        self.lambda_param = lambda_param
        # Liczba iteracji po wszystkich danych
        self.n_iters = n_iters
        # Wagi i bias, czyli parametry hiperpłaszczyzny dzielącej obiekty różnych klas
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if set(y) > {-1, 1}:
            raise Exception("SVM.fit accepts only -1 and 1 values of y")
        
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_iters):
            # Każdy rekord jest sprwadzany osobno. Wagi są delikatnie zbliżane do zerowego puntku gradientu dla danego punktu.
            for idx, x_i in (enumerate(X)):
                # Dla punktu leżącego po właściwej stronie marginesu, poszerzamy pas i odsuwamy od środka punktu 0 przez zmniejszego wag wektora "w"
                if y[idx] * (np.dot(self.w, x_i) - self.b) >= 1:
                    self.w -= self.lr * 2 * self.lambda_param * self.w 
                # Dla punktu leżącego po złej stronie marginesu: 
                # Wektor "w" zmieniany jest na dwa sposoby: 
                    # - zmniejszany dokładnie tyle samo ile w przypadku punktu po właściwej stronie
                    # - zmniejszamy lub zwiekszamy (w zależności od y) o część wektora x_i (czyli wskazującego w sprawdzany punkt)
                # Wartość "b" jest zmieniana tak, aby przesunąć hiperpłaszczyznę w kierunku przeciwnym do klasy sprawdzanego punktu, aby count wyszedł poza margines
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y[idx] * x_i) 
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        # Do predykcji mierzona jest odległość i zwrot punktu od linii, a następnie decyzja o klasyfikacji zależy od zwrotu  (ujemna czy dodatnia)
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)