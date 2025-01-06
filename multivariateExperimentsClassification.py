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

# Ignore all warnings
warnings.filterwarnings("ignore")

# Datasets
main_directory = "/home/filipe/Documentos/GitHub/LP_tests_estrutura_Arthur/Datasets/Selected_datasets"

datasets = MyPipeline.load_datasets(main_directory)

results_table = MyPipeline.create_results_table(datasets)
results_table_2 = results_table

list_algorithms = ["mean", "knn", "mice", "saei", "pmivae", "lpmd", "lpmd2"]

for algorithm in list_algorithms:

    # Parameters  
    n_metrics = 1
    imputation_model = algorithm  # mean, knn, mice, saei, pmivae, lpmd, lpmd2
    mechanism = "mar_median"  # mnar_mbouv, mnar_mbov_randomess, mar_correlated, mar_median
    approach = "Multivariate"

    results = {"dataset": [],
               "missing_rate": [],
               "accuracy": [],
               "recall": [],
               "precision": []}

    pipeline = MyPipeline()

    start = perf_counter()

    with open(f"./Analysis Results/Times/time_{imputation_model}.txt", "w") as file:
        # Generating results for the mechanism
        for data, name in zip(results_table["datasets"], results_table["dataset_names"]):
            for md in results_table["missing_rate"]:
                df = data.copy()
                missing_rate = md
                print(f"Dataset = {name} with MD = {md}")
                file.write(f"Dataset = {name} with MD = {md}\n")

                print("Testing Dataset Name: ")
                print("Dataset", name, "with MD =", md)

                cv = StratifiedKFold(n_splits=5)

                # Machine Learning (ML) model
                # model_classifier = DecisionTreeClassifier(random_state=42)
                model_classifier = RandomForestClassifier(max_depth=6, random_state=42)

                fold = 0

                X = df.drop(columns='target')
                y = df['target'].values
                x_cv = X.values

                general_mae_list = []

                # Cross-validation with 5 folds
                for train_index, test_index in cv.split(x_cv, y):
                    print("Fold = ", fold)
                    x_train, x_test = x_cv[train_index], x_cv[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    X_train = pd.DataFrame(x_train, columns=X.columns)
                    X_test = pd.DataFrame(x_test, columns=X.columns)

                    # Initializing the scaler
                    scaler = MyPipeline.initialize_scaler(X_train)

                    # Normalizing data
                    X_train_norm = MyPipeline.normalize_data(scaler, X_train)
                    X_test_norm = MyPipeline.normalize_data(scaler, X_test)

                    # Generating missing values independently for each set
                    
                    # MAR MEDIAN
                    imputation_md_train = mMAR(X=X_train_norm, y=y_train)
                    X_train_norm_md = imputation_md_train.median(missing_rate=md / 2)
                    X_train_norm_md = X_train_norm_md.drop(columns="target")

                    imputation_md_test = mMAR(X=X_test_norm, y=y_test)
                    X_test_norm_md = imputation_md_test.median(missing_rate=md / 2)
                    X_test_norm_md = X_test_norm_md.drop(columns="target")

                    start_imputation = perf_counter()
                    # Initializing and training the model
                    if imputation_model == "saei":
                        # SAEI
                        features = X_train_norm_md.columns[X_train_norm_md.isna().any()].tolist()
                        model = MyPipeline.choose_model(
                            model=imputation_model,
                            x_train=X_train_norm,
                            x_test=X_test_norm,
                            x_train_md=X_train_norm_md,
                            x_test_md=X_test_norm_md,
                            col_name=features,
                            input_shape=X.shape[1])

                        end_imputation = perf_counter()
                        file.write(f'Training time for fold = {fold} was = {end_imputation - start_imputation:.4f} s\n')

                        # Imputing missing values in training and test sets
                        output_md_train = model.transform(X_train_norm_md.iloc[:, :].values)
                        output_md_test = model.transform(X_test_norm_md.iloc[:, :].values)
                        
                    elif "lpmd" in imputation_model:
                        model = MyPipeline.choose_model(
                            model=imputation_model,
                            x_train=X_train_norm,
                            y_train=y_train,
                            x_train_md=X_train_norm_md)

                        end_imputation = perf_counter()
                        file.write(f'Training time for fold = {fold} was = {end_imputation - start_imputation:.4f} s\n')

                        # Imputing missing values in training and test sets
                        output_md_train = model.transform(X_train_norm_md.iloc[:, :].values, is_Train=True)
                        output_md_test = model.transform(X_test_norm_md.iloc[:, :].values)
                    
                    # KNN, MICE, PMIVAE, MEAN
                    else:
                        model = MyPipeline.choose_model(
                            model=imputation_model,
                            x_train=X_train_norm_md)

                        end_imputation = perf_counter()
                        file.write(f'Training time for fold = {fold} was = {end_imputation - start_imputation:.4f} s\n')

                        # Imputing missing values in training and test sets
                        output_md_train = model.transform(X_train_norm_md.iloc[:, :].values)
                        output_md_test = model.transform(X_test_norm_md.iloc[:, :].values)

                    X_imputado = np.concatenate([output_md_treino, output_md_test])
                    y_concat = np.concatenate([y_treino, y_teste])

                    # Save datasets to analyze the complexity of datasets with filled missing values
                    data_complex = pd.DataFrame(X_imputado.copy(), columns=X.columns)

                    data_complex["target"] = y_concat

                    os.makedirs(f"./Análises Resultados/Complexidade/{model_impt}/", exist_ok=True)

                    attributes = []
                    for j in data_complex:
                        if data_complex[j].dtypes in ['int64', 'float64', 'float32'] and j != "target":
                            attributes.append((j, 'NUMERIC'))
                        elif j == "target":
                            if nome == "bc_coimbra" or nome == "indian_liver":
                                attributes.append((j, ['1.0', '2.0']))
                            elif nome == "hcv_egyptian":
                                attributes.append((j, data_complex[j].unique().astype(str).tolist()))
                            else:
                                attributes.append((j, ['1.0', '0.0']))
                        else:
                            attributes.append((j, data_complex[j].unique().astype(str).tolist()))

                    dictarff = {'attributes': attributes,
                                'data': data_complex.values.tolist(),
                                'relation': f"{nome}"}

                    # Create ARFF file
                    arff_content = arff.dumps(dictarff)

                    # Save ARFF content to a file
                    with open(f"./Análises Resultados/Complexidade/{model_impt}/{model_impt}_fold{fold}_{nome}_{md}.arff", "w") as fcom:
                        fcom.write(arff_content)

                    # Fit the ML model
                    model_classifica.fit(output_md_treino, y_treino)

                    # Predict using the ML model
                    y_predict = model_classifica.predict(output_md_test)

                    results["dataset"].append(nome)
                    results["missing_rate"].append(md)
                    results['accuracy'].append(accuracy_score(y_teste, y_predict))
                    if nome == "hcv_egyptian":
                        results['recall'].append(recall_score(y_teste, y_predict, average="micro"))
                        results['precision'].append(precision_score(y_teste, y_predict, average="micro"))
                    else:
                        results['recall'].append(recall_score(y_teste, y_predict))
                        results['precision'].append(precision_score(y_teste, y_predict))

                    # Calculate MAE for the imputation on the test set
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
                    performance_mean = {"dataset": [],
                                        "missing_rate": [],
                                        "accuracy": [],
                                        "recall": [],
                                        "precision": []}

                    # Calculate accuracy, recall, and precision for classification
                    for step in range(0, len(performance), 5):
                        performance_mean["dataset"].append(results['dataset'][step:step+5][0])
                        performance_mean["missing_rate"].append(results['missing_rate'][step:step+5][0])
                        performance_mean["accuracy"].append(round(np.mean(results['accuracy'][step:step+5]), 4))
                        performance_mean["recall"].append(round(np.mean(results['recall'][step:step+5]), 4))
                        performance_mean["precision"].append(round(np.mean(results['precision'][step:step+5]), 4))

                    model_performance = pd.DataFrame(performance_mean)

                    model_performance.to_csv(f"./Análises Resultados/Classificação/{model_impt}_DecisionTree.csv")

                    resultados_final = MyPipeline.extrai_resultados(tabela_resultados)
                    resultados_final.to_csv(f"Resultados Instantaneos {abordagem}/{model_impt}_mnar-{mecanismo}.csv")

                    # Imputation results
                    resultados_mecanismo = MyPipeline.calcula_metricas_estatisticas_resultados(
                        resultados_final, n_metricas, fold
                    )
                    resultados_mecanismo.to_csv(
                        f"Resultados Parciais {abordagem}/{model_impt}_mnar-{mecanismo}.csv"
                    )

                    fim = perf_counter()
                    file.write(f'Total processing time for {model_impt.upper()} was = {fim-inicio:.4f} s')

                    print("Last algorithm: ", algorithm)


                    