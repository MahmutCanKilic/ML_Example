import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.metrics import accuracy_score
print("selam can")
data = pd.read_csv("/Users/salihcsr/Documents/GitHub/ML_Example/breast-cancer.csv")


data.info()
print("----------------")
print(data.describe())
print("----------------")
print(data.head(5))
print("----------------")


print(data.loc[30])
print("----------------")


Diagnosis_num = np.zeros((data.shape[0], 1)).astype(int)
for i in range(data.shape[0]):
    if data['diagnosis'][i] == 'M':
        Diagnosis_num[i] = 1

Diagnosis_num = pd.DataFrame(Diagnosis_num, columns=["Diagnosis_num"])
data.drop(['diagnosis'], axis=1, inplace=True)
data = pd.concat([Diagnosis_num, data], axis=1)

print("-----------------")
print(data.head(5))
print("----------------")


data_np = np.array(data)


random_siralama = np.random.permutation(data.shape[0])
data_np = data_np[random_siralama, :]


egitim_X = data_np[:int(data.shape[0] * 0.5), 1:]
egitim_y = data_np[:int(data.shape[0] * 0.5), 0]

val_X = data_np[int(data.shape[0] * 0.5):int(data.shape[0] * 0.7), 1:]
val_y = data_np[int(data.shape[0] * 0.5):int(data.shape[0] * 0.7), 0]

test_X = data_np[int(data.shape[0] * 0.7):, 1:]
test_y = data_np[int(data.shape[0] * 0.7):, 0]

print(egitim_X.shape, egitim_y.shape, val_X.shape, val_y.shape)
print("********************")

df = pd.DataFrame(data)
print(df.dtypes)
print("********************")

def uzaklik_hesapla(ornek, matris):
    uzakliklar = np.zeros(matris.shape[0])
    for i in range(matris.shape[0]):
        uzakliklar[i] = np.sqrt(np.sum((ornek - matris[i, :]) ** 2))
    return uzakliklar

def basari_hesapla(tahmin, gercek):
    t = 0
    for i in range(len(tahmin)):
        if tahmin[i] == gercek[i]:
            t += 1
    return (t / len(tahmin)) * 100

aday_k_lar = [1, 3, 5, 7, 9]

for k in aday_k_lar:
    tahminler = np.zeros(val_X.shape[0])
    for i in range(val_X.shape[0]):
        ornek = val_X[i, :]
        uzakliklar = uzaklik_hesapla(ornek, egitim_X)
        yakindan_uzaga_indisler = np.argsort(uzakliklar)
       
        mode_result = stats.mode(egitim_y[yakindan_uzaga_indisler[:k]])
        
        mode_value = mode_result.mode[0] if isinstance(mode_result.mode, np.ndarray) else mode_result.mode
        tahminler[i] = mode_value
    basari = basari_hesapla(tahminler, val_y)
    print(f'k= {k} için doğrulama başarısı: {basari}')