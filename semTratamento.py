# Declarando as bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import time


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



# Dividindo os valores de teste e de treino
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Treinando os modelos

tempo1begin = time.perf_counter()
modelo = ExtraTreesClassifier()
print("\n", modelo.fit(x_treino, y_treino))
tempo1end = time.perf_counter()


tempo2begin = time.perf_counter()
modelo2 = GaussianNB()
print("\n", modelo2.fit(x_treino, y_treino))
tempo2end = time.perf_counter()


tempo3begin = time.perf_counter()
modelo3 =  svm.SVC()
print("\n" , modelo3.fit(x_treino, y_treino))
tempo3end = time.perf_counter()


tempo1 = tempo1end - tempo1begin
tempo2 = tempo2end - tempo2begin
tempo3 = tempo3end - tempo3begin


# Imprimindo os valores
resultados = modelo.score(x_teste, y_teste)
resultados = resultados*100
print("\nPrecisão Arvore de busca: " + str(resultados) + " --  " + " com tempo de execução: " + str(tempo1))

resultados2 = modelo2.score(x_teste, y_teste)
resultados2 = resultados2*100
print("\nPrecisão Naive Bayes:" + str(resultados2) + " --  " + " com tempo de execução: " + str(tempo2))

resultados3 = modelo3.score(x_teste, y_teste)
resultados3 = resultados3*100
print("\nPrecisão SVM:" + str(resultados3) + " -- " + " com tempo de execução: " + str(tempo3) )