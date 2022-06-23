# Declarando as bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import svm

# Declarando o caminho do banco de dados
arquivo = pd.read_csv('heart.csv')

CAMPO_IDADE = "age" # Age
CAMPO_OUTPUT = 'output' # Target variable
CAMPO_TRTBPS = "trtbps" # Resting blood pressure (in mm Hg)
CAMPO_CHOL = "chol" # Cholestoral in mg/dl fetched via BMI sensor
CAMPO_THALACHH = "thalachh" # Maximum heart rate achieved
CAMPO_FBS = "fbs" #(fasting blood sugar > 120 mg/dl) ~ 1 = True, 0 = False
CAMPO_OLDPEAK = "oldpeak" 

# Tratando os dados
arquivo[CAMPO_OUTPUT] = arquivo[CAMPO_OUTPUT].replace('Sem chance', 0)
arquivo[CAMPO_OUTPUT] = arquivo[CAMPO_OUTPUT].replace('Com chance', 1)


print("\n", arquivo.head())

y = arquivo[CAMPO_OUTPUT]
x = arquivo.drop(columns=[CAMPO_CHOL, CAMPO_FBS, CAMPO_OUTPUT], axis=1)

# Diminuindo os outliers
arquivo[CAMPO_IDADE] = np.log(arquivo.age)
arquivo[CAMPO_TRTBPS] = np.log(arquivo.trtbps)
arquivo[CAMPO_CHOL] = np.log(arquivo.chol)
arquivo[CAMPO_THALACHH] = np.log(arquivo.thalachh)

colunas_continua = [CAMPO_IDADE, CAMPO_TRTBPS, CAMPO_CHOL, CAMPO_THALACHH, CAMPO_OLDPEAK]
dados_continuos = arquivo[colunas_continua]

for k, v in dados_continuos.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    #Definição de valores limites para validar se é ou não um outlier
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(arquivo)[0]
    print("Column {} outliers = {} => {}%".format(k, len(v_col), round(perc, 3)))

# Dividindo os valores de teste e de treino
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Treinando os modelos
modelo = ExtraTreesClassifier()
print("\n", modelo.fit(x_treino, y_treino))

modelo2 = GaussianNB()
print("\n", modelo2.fit(x_treino, y_treino))

modelo3 =  svm.SVC()
print("\n" , modelo3.fit(x_treino, y_treino))

# Imprimindo os valores
resultados = modelo.score(x_teste, y_teste)
resultados = resultados*100
print("\nPrecisão Arvore de busca: " + str(resultados) + "%")

resultados2 = modelo2.score(x_teste, y_teste)
resultados2 = resultados2*100
print("\nPrecisão Naive Bayes:" + str(resultados2) + "%")

resultados3 = modelo3.score(x_teste, y_teste)
resultados3 = resultados3*100
print("\nPrecisão SVM:" + str(resultados3) + "%")