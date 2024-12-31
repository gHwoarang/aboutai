import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Veri kümesi oluşturma
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Yapay sinir ağı modeli tanımlama
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Giriş katmanı: 2 giriş, 4 gizli nöron
        self.fc2 = nn.Linear(4, 1)  # Çıkış katmanı: 4 gizli nöron, 1 çıkış

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Aktivasyon fonksiyonu olarak sigmoid kullanıyoruz
        x = torch.sigmoid(self.fc2(x))
        return x

# Modeli oluşturma
model = NeuralNetwork()

# Hata fonksiyonu (kayıp fonksiyonu) ve optimize edici tanımlama
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Eğitim döngüsü
epochs = 1000
for epoch in range(epochs):
    # Önceki gradyanları sıfırla
    optimizer.zero_grad()
    # İleri yayılım
    outputs = model(X)
    # Hesaplanan hata
    loss = criterion(outputs, y)
    # Geriye doğru yayılım
    loss.backward()
    # Parametreleri güncelle
    optimizer.step()

    # Her 100 epoch'ta bir hata yazdır
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Eğitilmiş modeli test etme
with torch.no_grad():
    outputs = model(X)
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == y).float().mean()
    print(f'Accuracy: {accuracy.item()*100:.2f}%')


