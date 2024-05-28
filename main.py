import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from flask import Flask, request, render_template

main = Flask(__name__)

data = pd.read_csv("/uygulama/breast-cancer.csv")

data = data.drop(columns=["id"])

Diagnosis_num = np.zeros((data.shape[0], 1)).astype(int)
for i in range(data.shape[0]):
    if data['diagnosis'][i] == 'M':
        Diagnosis_num[i] = 1

Diagnosis_num = pd.DataFrame(Diagnosis_num, columns=["Diagnosis_num"])
data.drop(['diagnosis'], axis=1, inplace=True)
data = pd.concat([Diagnosis_num, data], axis=1)

data_np = np.array(data)
random_siralama = np.random.permutation(data.shape[0])
data_np = data_np[random_siralama, :]

egitim_X = data_np[:int(data.shape[0] * 0.5), 1:]
egitim_y = data_np[:int(data.shape[0] * 0.5), 0]

val_X = data_np[int(data.shape[0] * 0.5):int(data.shape[0] * 0.7), 1:]
val_y = data_np[int(data.shape[0] * 0.5):int(data.shape[0] * 0.7), 0]

test_X = data_np[int(data.shape[0] * 0.7):, 1:]
test_y = data_np[int(data.shape[0] * 0.7):, 0]

df = pd.DataFrame(data)
scaler = MinMaxScaler()
egitim_X = scaler.fit_transform(egitim_X)
val_X = scaler.transform(val_X)
test_X = scaler.transform(test_X)

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

def knn_predict(egitim_X, egitim_y, test_X, k):
    tahminler = np.zeros(test_X.shape[0])
    for i in range(test_X.shape[0]):
        ornek = test_X[i, :]
        uzakliklar = uzaklik_hesapla(ornek, egitim_X)
        yakindan_uzaga_indisler = np.argsort(uzakliklar)
        mode_result = stats.mode(egitim_y[yakindan_uzaga_indisler[:k]])
        mode_value = mode_result.mode[0] if isinstance(mode_result.mode, np.ndarray) else mode_result.mode
        tahminler[i] = mode_value
    return tahminler

def naive_bayes_predict(train_X, train_y, test_X):
    clf = GaussianNB()
    clf.fit(train_X, train_y)
    return clf.predict(test_X)

def logistic_regression_predict(train_X, train_y, test_X):
    clf = LogisticRegression()
    clf.fit(train_X, train_y)
    return clf.predict(test_X)

aday_k_lar = [1, 3, 5, 7, 9]
best_k = None
best_accuracy = 0
validation_accuracies_knn = []

for k in aday_k_lar:
    tahminler = knn_predict(egitim_X, egitim_y, val_X, k)
    basari = basari_hesapla(tahminler, val_y)
    print(f'k= {k} için doğrulama başarısı (KNN): {basari}')
    validation_accuracies_knn.append(basari)
    if basari > best_accuracy:
        best_accuracy = basari
        best_k = k

print(f'En iyi k (KNN): {best_k} ile doğrulama başarısı: {best_accuracy}')


naive_bayes_predictions = naive_bayes_predict(egitim_X, egitim_y, val_X)
naive_bayes_accuracy = accuracy_score(val_y, naive_bayes_predictions)
print(f'Naive Bayes doğrulama başarısı: {naive_bayes_accuracy}')


logistic_regression_predictions = logistic_regression_predict(egitim_X, egitim_y, val_X)
logistic_regression_accuracy = accuracy_score(val_y, logistic_regression_predictions)
print(f'Lojistik Regresyon doğrulama başarısı: {logistic_regression_accuracy}')

# Permütasyon Önemi için Kod
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(egitim_X, egitim_y)
perm_importance = permutation_importance(logistic_regression_model, val_X, val_y, n_repeats=10, random_state=42)
feature_importance = perm_importance.importances_mean
feature_names = data.columns[1:]

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    final_features = scaler.transform(final_features)
    
    tahminler = knn_predict(egitim_X, egitim_y, final_features, best_k)
    mode_value = tahminler[0]
    if mode_value == 1:
        result = "Malignant"
    else:
        result = "Benign"
        
    return render_template('index.html', prediction_text=f'Cancer Type Prediction: {result}')    

plt.figure(figsize=(8, 6))
plt.barh(feature_names, feature_importance, color='lightblue')
plt.xlabel('Permütasyon Önemi')
plt.ylabel('Özellikler')
plt.title('Özellik Önemi (Lojistik Regresyon)')
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(14, 6))  

plt.subplot(1, 3, 1)
plt.suptitle("Grafikler")
sns.barplot(x=aday_k_lar, y=validation_accuracies_knn, palette="viridis")
plt.title('KNN Doğrulama Başarısı vs. K Değeri (Barplot)')
plt.xlabel('K Değeri')
plt.ylabel('Doğrulama Başarısı (%)')
plt.grid(True)
plt.ylim(80, 100)


plt.subplot(1, 3, 2)
plt.bar(['Naive Bayes'], [naive_bayes_accuracy * 100], color='skyblue')
plt.title('Naive Bayes Doğrulama Başarısı')
plt.xlabel('Model')
plt.ylabel('Doğrulama Başarısı (%)')
plt.grid(True)
plt.ylim(80, 100)


plt.subplot(1, 3, 3)
plt.bar(['Lojistik Regresyon'], [logistic_regression_accuracy * 100], color='salmon')
plt.title('Lojistik Regresyon Doğrulama Başarısı')
plt.xlabel('Model')
plt.ylabel('Doğrulama Başarısı (%)')
plt.grid(True)
plt.ylim(80, 100)

plt.tight_layout() 
plt.show()

if __name__ == "__main__":
    main.run(debug=True, port=5001)
