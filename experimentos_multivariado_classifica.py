from MyUtils import MyPipeline
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from time import perf_counter
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
import arff
import warnings
import os
from sklearn.ensemble import RandomForestClassifier

from mdatagen.multivariate.mMNAR import mMNAR
from mdatagen.multivariate.mMAR import mMAR

# Ignorar todos os avisos
warnings.filterwarnings("ignore")

# Datasets
diretorio_principal = "/home/filipe/Documentos/GitHub/LP_tests_estrutura_Arthur/Datasets/Selected_datasets"

datasets = MyPipeline.carrega_datasets(diretorio_principal)

tabela_resultados = MyPipeline.cria_tabela_resultados(datasets)
tabela_resultados_2 = tabela_resultados

list_algorithms = ["mean", "knn", "mice", "saei", "pmivae", "lpr0", "lpr1", "lpr2", "lpr3"]
#list_algorithms = ["mean", "knn", "mice", "saei", "pmivae", "lpr0", "lpr1", "lpr2", "lpr3"]

for algorithm in list_algorithms:

    # Parâmetros  
    n_metricas = 1
    model_impt = algorithm # mean, knn, mice, saei, pmivae, lpmd, lpmd2
    mecanismo = "mar_median" # mnar_mbouv, mnar_mbov_randomess, mar_correlated, mar_median
    abordagem = "Multivariado"


    results = {"dataset":[],
            "missing_rate": [],
                "accuracy":[],
                "recall": [],
                "precision": []}

    pipeline = MyPipeline()

    inicio = perf_counter()

    with open(f"./Análises Resultados/Tempos/tempo_{model_impt}.txt", "w") as file:
        # Gerando resultados para os mecanismo
        for dados, nome in zip(tabela_resultados["datasets"], tabela_resultados["nome_datasets"]):
            for md in tabela_resultados["missing_rate"]:
                df = dados.copy()
                missing_rate = md
                print(f"Dataset = {nome} com MD = {md}")
                file.write(f"Dataset = {nome} com MD = {md}\n")

                print("Teste Nome Dataset: ")
                print("Dataset", nome," com MD =", md)

                cv = StratifiedKFold(n_splits=5)

                # Modelo de Machine Learning (ML)
                #model_classifica = DecisionTreeClassifier(random_state=42)
                model_classifica = RandomForestClassifier(max_depth=6, random_state=42)
                
                fold = 0

                X = df.drop(columns = 'target')
                y = df['target'].values
                x_cv = X.values

                lista_maes_geral = []

                # Cross-validation with 5 folds
                for train_index, test_index in cv.split(x_cv, y):
                    print("Fold = ", fold)
                    x_treino, x_teste = x_cv[train_index], x_cv[test_index]
                    y_treino, y_teste = y[train_index], y[test_index]

                    X_treino = pd.DataFrame(x_treino, columns=X.columns)
                    X_teste = pd.DataFrame(x_teste, columns=X.columns)

                    # Inicializando o normalizador (scaler)
                    scaler = MyPipeline.inicializa_normalizacao(X_treino)

                    # Normalizando os dados
                    X_treino_norm = MyPipeline.normaliza_dados(scaler, X_treino)
                    X_teste_norm = MyPipeline.normaliza_dados(scaler, X_teste)

                    # Geração dos missing values em cada conjunto de forma independente
                    
                    # MNAR MBOUV
                    #impt_md_train = mMNAR(X=X_treino_norm, y=y_treino)
                    #X_treino_norm_md = impt_md_train.MBOUV(missing_rate=md,depend_on_external=X.columns, ascending=True)
                    #X_treino_norm_md = X_treino_norm_md.drop(columns="target")

                    #impt_md_test = mMNAR(X=X_teste_norm, y = y_teste)
                    #X_teste_norm_md = impt_md_test.MBOUV(missing_rate=md,depend_on_external=X.columns, ascending=True)
                    #X_teste_norm_md = X_teste_norm_md.drop(columns="target")

                    # MNAR MBOV randomess
                    # impt_md_train = mMNAR(X=X_treino_norm, y=y_treino)
                    # X_treino_norm_md = impt_md_train.MBOV_randomness(missing_rate=md, randomness=0.5, columns=X.columns)
                    # X_treino_norm_md = X_treino_norm_md.drop(columns="target")

                    # impt_md_test = mMNAR(X=X_teste_norm, y = y_teste)
                    # X_teste_norm_md = impt_md_test.MBOV_randomness(missing_rate=md, randomness=0.5, columns=X.columns)
                    # X_teste_norm_md = X_teste_norm_md.drop(columns="target")

                    
                    # MNAR Random (Deu erro)
                    # impt_md_train = mMNAR(X=X_treino_norm, y=y_treino)
                    # X_treino_norm_md = impt_md_train.random(missing_rate=md)
                    # X_treino_norm_md = X_treino_norm_md.drop(columns="target")

                    # impt_md_test = mMNAR(X=X_teste_norm, y = y_teste)
                    # X_teste_norm_md = impt_md_test.random(missing_rate=md)
                    # X_teste_norm_md = X_teste_norm_md.drop(columns="target")

                    # MAR Random - MESMO ERRO QUE O MNAR RANDOM
                    #verifica o numero de colunas
                    # colunas = X_treino.columns
                    # quant_colunas = len(colunas)

                    # impt_md_train = mMAR(X=X_treino_norm, y=y_treino, n_xmiss=quant_colunas)
                    # X_treino_norm_md = impt_md_train.random(missing_rate=md)
                    # X_treino_norm_md = X_treino_norm_md.drop(columns="target")

                    # impt_md_test = mMAR(X=X_teste_norm, y = y_teste, n_xmiss=quant_colunas)
                    # X_teste_norm_md = impt_md_test.random(missing_rate=md)
                    # X_teste_norm_md = X_teste_norm_md.drop(columns="target")

                    #MAR CORRELATED
                    # impt_md_train = mMAR(X=X_treino_norm, y=y_treino)
                    # X_treino_norm_md = impt_md_train.correlated(missing_rate=md/2)
                    # X_treino_norm_md = X_treino_norm_md.drop(columns="target")

                    # impt_md_test = mMAR(X=X_teste_norm, y=y_teste)
                    # X_teste_norm_md = impt_md_test.correlated(missing_rate=md/2)
                    # X_teste_norm_md = X_teste_norm_md.drop(columns="target")

                    #MAR MEDIAN
                    impt_md_train = mMAR(X=X_treino_norm, y=y_treino)
                    X_treino_norm_md = impt_md_train.median(missing_rate=md/2)
                    X_treino_norm_md = X_treino_norm_md.drop(columns="target")

                    impt_md_test = mMAR(X=X_teste_norm, y = y_teste)
                    X_teste_norm_md = impt_md_test.median(missing_rate=md/2)
                    X_teste_norm_md = X_teste_norm_md.drop(columns="target")


                    inicio_imputation = perf_counter()
                    # Inicializando e treinando o modelo
                    if model_impt == "saei":
                        # SAEI
                        features = X_treino_norm_md.columns[X_treino_norm_md.isna().any()].tolist()
                        model = MyPipeline.choose_model(model = model_impt, 
                                                        x_train = X_treino_norm, 
                                                        x_test = X_teste_norm,
                                                        x_train_md = X_treino_norm_md,
                                                        x_test_md = X_teste_norm_md,
                                                        col_name =  features,
                                                        input_shape = X.shape[1])
                        
                        fim_imputation = perf_counter()
                        file.write(f'Tempo de treinamento para fold = {fold} foi = {fim_imputation-inicio_imputation:.4f} s\n')

                        # Imputação dos missing values nos conjuntos de treino e teste
                        output_md_treino = model.transform(X_treino_norm_md.iloc[:, :].values)
                        output_md_test = model.transform(X_teste_norm_md.iloc[:, :].values)
                        
                    elif "lpr" in model_impt:
                        # LABEL PROPAGATION REGRESSION 0, 1, 2, 3
                        model = MyPipeline.choose_model(model = model_impt, 
                                                        x_train = X_treino_norm,
                                                        y_train = y_treino, 
                                                        x_train_md = X_treino_norm_md)
                        
                        fim_imputation = perf_counter()
                        file.write(f'Tempo de treinamento para fold = {fold} foi = {fim_imputation-inicio_imputation:.4f} s\n')

                        # Imputação dos missing values nos conjuntos de treino e teste
                        output_md_treino = model.transform(X_treino_norm_md.iloc[:, :].values, is_Train=True)
                        output_md_test = model.transform(X_teste_norm_md.iloc[:, :].values)
                    
                    # KNN, MICE, PMIVAE, MEAN
                    else:
                        model = MyPipeline.choose_model(
                            model=model_impt,
                            x_train=X_treino_norm_md,)
                        
                        fim_imputation = perf_counter()
                        file.write(f'Tempo de treinamento para fold = {fold} foi = {fim_imputation-inicio_imputation:.4f} s\n')

                        # Imputação dos missing values nos conjuntos de treino e teste
                        output_md_treino = model.transform(X_treino_norm_md.iloc[:, :].values)
                        output_md_test = model.transform(X_teste_norm_md.iloc[:, :].values)


                    X_imputado = np.concatenate([output_md_treino, output_md_test])
                    y_concat = np.concatenate([y_treino, y_teste])


                    # Salva os datasets para efetuar análise de complexidade dos datasets com missing preenchidos
                    data_complex = pd.DataFrame(X_imputado.copy(), columns = X.columns)

                    data_complex["target"] = y_concat

                    os.makedirs(f"./Análises Resultados/Complexidade/{model_impt}/", exist_ok=True)  

                    attributes = []
                    for j in data_complex:
                        if data_complex[j].dtypes in ['int64', 'float64', 'float32'] and j != "target":
                            attributes.append((j, 'NUMERIC'))
                        elif j == "target":
                            if nome == "bc_coimbra" or nome == "indian_liver":
                                attributes.append((j, ['1.0','2.0']))
                            elif nome == "hcv_egyptian":
                                attributes.append((j, data_complex[j].unique().astype(str).tolist()))
                            else:
                                attributes.append((j, ['1.0','0.0']))
                        else:
                            attributes.append((j, data_complex[j].unique().astype(str).tolist()))


                    dictarff = {'attributes': attributes,
                                'data': data_complex.values.tolist(),
                                'relation': f"{nome}"}

                    # Criar o arquivo ARFF
                    arff_content = arff.dumps(dictarff)

                    # Salvar o conteúdo ARFF em um arquivo
                    with open(f"./Análises Resultados/Complexidade/{model_impt}/{model_impt}_fold{fold}_{nome}_{md}.arff", "w") as fcom:
                        fcom.write(arff_content)

                    # Fit do modelo de ML
                    model_classifica.fit(output_md_treino,y_treino)
                    
                    # Predict do modelo de ML
                    y_predict = model_classifica.predict(output_md_test)

                    results["dataset"].append(nome)
                    results["missing_rate"].append(md)
                    results['accuracy'].append(accuracy_score(y_teste,y_predict))
                    if nome == "hcv_egyptian":
                        results['recall'].append(recall_score(y_teste,y_predict, average="micro"))
                        results['precision'].append(precision_score(y_teste,y_predict, average="micro"))
                    else:
                        results['recall'].append(recall_score(y_teste,y_predict))
                        results['precision'].append(precision_score(y_teste,y_predict))
                    
                    # Calculando MAE para a imputação no conjunto de teste
                    mae_teste_mean, mae_teste_std, lista_maes = MyPipeline.gera_resultado_multiva(
                        resposta=output_md_test,
                        dataset_normalizado_md=X_teste_norm_md,
                        dataset_normalizado_original=X_teste_norm,
                    )

                    lista_maes_geral.append(lista_maes)

                    tabela_resultados[f"{model_impt}/{nome}/{md}/{fold}/MAE"] = {
                        "teste": mae_teste_mean
                    }

                    fold += 1

        performance = pd.DataFrame(results)
        performance_mean = {"dataset":[],
                            "missing_rate": [],
                            "accuracy":[],
                            "recall": [],
                            "precision": []}

        # Calculando acurácia, recall e precisão para classificação 
        for step in range(0,len(performance),5):
            performance_mean["dataset"].append(results['dataset'][step:step+5][0])
            performance_mean["missing_rate"].append(results['missing_rate'][step:step+5][0])
            performance_mean["accuracy"].append(round(np.mean(results['accuracy'][step:step+5]),4))
            performance_mean["recall"].append(round(np.mean(results['recall'][step:step+5]),4))
            performance_mean["precision"].append(round(np.mean(results['precision'][step:step+5]),4))

        model_performance = pd.DataFrame(performance_mean)

        model_performance.to_csv(f"./Análises Resultados/Classificação/{model_impt}_DecisionTree.csv")

        resultados_final = MyPipeline.extrai_resultados(tabela_resultados)
        resultados_final.to_csv(f"Resultados Instantaneos {abordagem}/{model_impt}_mnar-{mecanismo}.csv")

        # Resultados da imputação
        resultados_mecanismo = MyPipeline.calcula_metricas_estatisticas_resultados(
            resultados_final, n_metricas, fold
        )
        resultados_mecanismo.to_csv(
            f"Resultados Parciais {abordagem}/{model_impt}_mnar-{mecanismo}.csv"
        )

        fim = perf_counter()
        file.write(f'Tempo de total de processamento total para {model_impt.upper()} foi = {fim-inicio:.4f} s')

        print("Último algoritmo: ", algorithm)

