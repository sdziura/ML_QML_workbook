from torch import nn

class Two_Layer_Model(nn.Module):
    def __init__(self):
        super(Two_Layer_Model, self).__init__() # Inicjalizacja dziedziczonego nn.Module
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