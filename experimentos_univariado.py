from MyUtils import MyPipeline
from missing_data_generator import MissingDataGenerator
from sklearn.model_selection import KFold
import pandas as pd
from time import perf_counter

# Datasets
diretorio_principal = "../Datasets/"

datasets = MyPipeline.carrega_datasets(diretorio_principal)

tabela_resultados = MyPipeline.cria_tabela_resultados(datasets)

n_metricas = 1
model_impt = "saei"
mecanismo = 0
k = 0.2  # parâmetro para VAE-BRIDGE

pipeline = MyPipeline()
inicio = perf_counter()

# Gerando resultados para os mecanismo
for dados, nome in zip(tabela_resultados["datasets"], tabela_resultados["nome_datasets"]):
    for md in tabela_resultados["missing_rate"]:
        df = dados.copy()
        missing_rate = md
        print(f"Dataset = {nome} com MD = {md}")

        cv = KFold(n_splits=5)
        fold = 0

        X = df.drop(columns = 'target')
        y = df['target'].values
        x_cv = X.values

        # Cross-validation with 5 folds
        for train_index, test_index in cv.split(x_cv):
            print("Fold = ", fold)
            x_treino, x_teste = x_cv[train_index], x_cv[test_index]
            y_treino, y_teste = y[train_index], y[test_index]

            X_treino = pd.DataFrame(x_treino, columns=X.columns)
            X_teste = pd.DataFrame(x_teste, columns=X.columns)

            # Normalizando os dados
            scaler = MyPipeline.inicializa_normalizacao(X_treino)

            X_treino_norm = MyPipeline.normaliza_dados(scaler, X_treino)
            X_teste_norm = MyPipeline.normaliza_dados(scaler, X_teste)

            # Imputação dos missing values em cada conjunto
            impt_md = MissingDataGenerator(X=X, y=y, missing_rate=missing_rate)
            X_treino_norm_md = impt_md.MNAR_univa(X_treino_norm, mecanismo)
            X_teste_norm_md = impt_md.MNAR_univa(X_teste_norm, mecanismo)

            feature = X_treino_norm_md.columns[X_treino_norm_md.isna().any()].tolist()[0]
            missing_id = X_treino_norm_md.columns.get_loc(feature)

            # Inicializando e treinando o modelo
            # model = MyPipeline.choose_model(
            #     model=model_impt,
            #     x_train=X_treino_norm_md,
            #     k_perc = k, 
            #     col_name =  feature,
            #     missing_feature_id = missing_id,)
            
            # SAEI
            model = MyPipeline.choose_model(
                model=model_impt,
                x_train=X_treino_norm,
                x_test= X_teste_norm,
                x_train_md = X_treino_norm_md,
                x_test_md = X_teste_norm_md,
                col_name =  feature,
                missing_feature_id = missing_id,
                input_shape = X.shape[1])
            

            # No conjunto de teste
            mae_teste = MyPipeline.gera_resultado_univa(
                model_fitted=model,
                dataset_normalizado_md=X_teste_norm_md,
                dataset_normalizado_original=X_teste_norm,
                missing_id = missing_id
            )

            tabela_resultados[f"{model_impt}/{nome}/{md}/{fold}/MAE"] = {
                "teste": mae_teste
            }

            fold += 1

fim = perf_counter()

resultados_final = MyPipeline.extrai_resultados(tabela_resultados)

resultados_mecanismo = MyPipeline.calcula_metricas_estatisticas_resultados(
    resultados_final, n_metricas, fold
)
resultados_mecanismo.to_csv(
    f"Resultados Parciais Univariado/{model_impt}_mnar-{mecanismo}.csv"
)

print(f'Tempo de processamento para {model_impt.upper()} foi = {fim-inicio:.4f} s')