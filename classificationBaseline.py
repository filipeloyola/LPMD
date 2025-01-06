from MyUtils import MyPipeline
from sklearn.model_selection import StratifiedKFold
from time import perf_counter
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import arff

# Datasets
#diretorio_principal = "C:\\Users\\Mult-e\\Desktop\\@Codigos\\Datasets\\"
diretorio_principal = "/home/filipe/Documentos/GitHub/LP_tests_estrutura_Arthur/Datasets/Selected_datasets"


datasets = MyPipeline.carrega_datasets(diretorio_principal)

tabela_resultados = MyPipeline.cria_tabela_resultados(datasets)


#model_classify = 'DecisionTree'
model_classify = 'RandomForest'

inicio = perf_counter()

results = {"dataset":[],
            "accuracy":[],
            "recall": [],
            "precision": []}

# Gerando resultados da classificação para os datasets baseline
for dados, nome in zip(tabela_resultados["datasets"], tabela_resultados["nome_datasets"]):
    df = dados.copy()
    print(f"Dataset = {nome}")

    cv = StratifiedKFold(n_splits=5)
    #model = DecisionTreeClassifier(random_state=42) # colocar random forest
    model = RandomForestClassifier(max_depth=6, random_state=42)

    fold = 0

    X = df.drop(columns = 'target')
    y = df['target'].values
    x_cv = X.values

    # Stratified Cross-validation with 5 folds
    for train_index, test_index in cv.split(x_cv, y):
        print("Fold = ", fold)
        x_treino, x_teste = x_cv[train_index], x_cv[test_index]
        y_treino, y_teste = y[train_index], y[test_index]

        X_treino = pd.DataFrame(x_treino, columns=X.columns)
        X_teste = pd.DataFrame(x_teste, columns=X.columns)

        # Normalizando os dados
        scaler = MyPipeline.inicializa_normalizacao(X_treino)

        X_treino_norm = MyPipeline.normaliza_dados(scaler, X_treino)
        X_teste_norm = MyPipeline.normaliza_dados(scaler, X_teste)

        model.fit(X_treino, y_treino)

        y_predict = model.predict(X_teste)

        results["dataset"].append(nome)
        results['accuracy'].append(accuracy_score(y_teste,y_predict))
        if nome == "hcv_egyptian":
            results['recall'].append(recall_score(y_teste,y_predict, average="micro"))
            results['precision'].append(precision_score(y_teste,y_predict, average="micro"))
        else:
            results['recall'].append(recall_score(y_teste,y_predict))
            results['precision'].append(precision_score(y_teste,y_predict))

        fold += 1

fim = perf_counter()
    
performance = pd.DataFrame(results)
performance_mean = {"dataset":[],
            "accuracy":[],
            "recall": [],
            "precision": []}

for step in range(0,len(performance),5):
    performance_mean["dataset"].append(results['dataset'][step:step+5][0])
    performance_mean["accuracy"].append(round(np.mean(results['accuracy'][step:step+5]),4))
    performance_mean["recall"].append(round(np.mean(results['recall'][step:step+5]),4))
    performance_mean["precision"].append(round(np.mean(results['precision'][step:step+5]),4))

model_performance = pd.DataFrame(performance_mean)


model_performance.to_csv(f"./Análises Resultados/Classificação/{model_classify}_baseline.csv")


print(f'Tempo de processamento para {model_classify} foi = {fim-inicio:.4f} s')
