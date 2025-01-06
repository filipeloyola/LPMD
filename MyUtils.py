import os
import pandas as pd
from scipy.io import arff
from io import StringIO

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Label Propagation Regression
from LPMD import *

# Variational Autoencoder Filter for Bayesian Ridge Imputation
from Autoencoders.bridge import VAEBRIDGE
from Autoencoders.vae_bridge import ConfigVAE

# MICE, KNN, Dumb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


# Partial Multiple Imputation with Variational Autoencoders
from Autoencoders.pmivae import PMIVAE
from Autoencoders.vae_pmivae import ConfigVAE

# Siamese Autoencoder
from Autoencoders.saei import ConfigSAE, SAEImp, DataSets

# Bibliotecas
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


from pycol import Complexity
import warnings

# Ignorar todos os avisos
warnings.filterwarnings("ignore")


# ==========================================================================
class MyPipeline:
    # ------------------------------------------------------------------------
    def label_encoder(df, lista_nome_colunas):
        data = df.copy()
        le = LabelEncoder()

        for att in lista_nome_colunas:
            data[att] = le.fit_transform(data[att])

        return data

    # ------------------------------------------------------------------------
    def one_hot_encode(df, lista_nome_colunas):
        data = df.copy()
        data_categorical = data[lista_nome_colunas]

        encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = encoder.fit_transform(data_categorical)

        # Criar um novo DataFrame com as colunas one-hot
        columns = encoder.get_feature_names_out(lista_nome_colunas)
        df_encoded = pd.DataFrame(one_hot_encoded, columns=columns)

        # Concatenar o DataFrame original com o DataFrame codificado
        data = pd.concat([data, df_encoded], axis=1)

        # Remover as colunas originais
        data = data.drop(columns=lista_nome_colunas)

        return data

    # ------------------------------------------------------------------------
    def cria_tabela_resultados(datasets):
        tabela_resultados = {}

        (
            breast_cancer_wisconsin_df,
            pima_diabetes_df,
            bc_coimbra_df,
            indian_liver_df,
            parkinsons_df,
            mammographic_masses_df,
            hcv_egyptian_df,
            thoracic_surgery_df,
            # cdc_diabetes_df
        ) = MyPipeline.pre_processing_datasets(datasets)

        tabela_resultados["datasets"] = [
            breast_cancer_wisconsin_df,
            pima_diabetes_df,
            bc_coimbra_df,
            indian_liver_df,
            parkinsons_df,
            mammographic_masses_df,
            hcv_egyptian_df,
            thoracic_surgery_df,
            # cdc_diabetes_df
        ]
        tabela_resultados["nome_datasets"] = [
            "wiscosin",
            "pima_diabetes",
            "bc_coimbra",
            "indian_liver",
            "parkinsons",
            "mammographic_masses",
            "hcv_egyptian",
            "thoracic_surgey",
            # "cdc_diabetes"
        ]
        tabela_resultados["missing_rate"] = [10, 20, 40, 60]

        return tabela_resultados

    # ------------------------------------------------------------------------
    def pre_processing_datasets(datasets: dict):
        breast_cancer_wisconsin_df = datasets["wiscosin"].copy()
        breast_cancer_wisconsin_df = breast_cancer_wisconsin_df.drop(columns="ID")
        breast_cancer_wisconsin_df = MyPipeline.label_encoder(
            breast_cancer_wisconsin_df, ["target"]
        )

        pima_diabetes_df = datasets["pima_diabetes"].copy()

        bc_coimbra_df = datasets["bc_coimbra"].copy()

        indian_liver_df = datasets["indian_liver"].copy()
        indian_liver_df = indian_liver_df.dropna()
        indian_liver_df = MyPipeline.label_encoder(indian_liver_df, ["Gender"])

        datasets["parkinsons"] = datasets["parkinsons"].drop(columns="name")
        parkinsons_df = datasets["parkinsons"].copy()

        mammographic_masses_df = datasets["mammographic_masses"].copy()
        mammographic_masses_df = (
            datasets["mammographic_masses"]
            .replace("?", np.nan)
            .dropna()
            .drop(columns="BI-RADS assessment")
            .reset_index(drop=True)
        )
        mammographic_masses_df["Age"] = mammographic_masses_df["Age"].astype("int64")
        mammographic_masses_df["Density"] = mammographic_masses_df["Density"].astype(
            "int64"
        )
        mammographic_masses_df = MyPipeline.one_hot_encode(
            mammographic_masses_df, ["Shape", "Margin"]
        )

        hcv_egyptian_df = datasets["HCV-Egy-Data"].copy()

        thoracic_surgery_df = datasets["ThoraricSurgery"].copy()
        thoracic_surgery_df = MyPipeline.label_encoder(
            MyPipeline.one_hot_encode(thoracic_surgery_df, ["DGN"]),
            [
                "PRE6",
                "PRE7",
                "PRE8",
                "PRE9",
                "PRE10",
                "PRE11",
                "PRE14",
                "PRE17",
                "PRE19",
                "PRE25",
                "PRE30",
                "PRE32",
                "target",
            ],
        )
        cdc_diabetes_df = datasets["cdc_diabetes"].copy()

        return (
            breast_cancer_wisconsin_df,
            pima_diabetes_df,
            bc_coimbra_df,
            indian_liver_df,
            parkinsons_df,
            mammographic_masses_df,
            hcv_egyptian_df,
            thoracic_surgery_df,
            # cdc_diabetes_df
        )

    # ------------------------------------------------------------------------
    def cria_dataframe(df) -> pd.DataFrame:
        """
        Função para criar um pandas DataFrame a partir de datasets da biblioteca do sklearn.datasets

        Args:
            df: Um objeto pandas DataFrame.

        Returns:
            Um objeto pandas DataFrame contendo os dados do DataFrame de entrada (df) com uma coluna adicional chamada 'target'.
        """
        dataset = pd.DataFrame(data=df.data, columns=df.feature_names)
        dataset["target"] = df.target
        return dataset

    # ------------------------------------------------------------------------
    def split_dataset(dataset: pd.DataFrame, perc_treino: float, perc_teste: float):
        """
        Divide o dataset dado nos conjunto de treino, teste e validação.

        Args:
            dataset (pd.DataFrame): A pandas DataFrame contendo o dataset a ser dividido.
            perc_treino (float): A porcentagem do dataset que será usada para treinamento
            perc_teste (float): A porcentagem do dataset que será usada para teste.

        Returns:
            tuple: A tuple contendo três numpy arrays: (X_treino, X_teste, X_valida)
        """

        dataset = dataset.copy()
        df_shuffle = dataset.sample(frac=1.0, replace=True)

        tamanho_treino = int(perc_treino * len(dataset))
        tamanho_teste = int(perc_teste * len(dataset))

        x_treino = df_shuffle.iloc[:tamanho_treino]
        x_teste = df_shuffle.iloc[tamanho_treino : tamanho_treino + tamanho_teste]
        x_valida = df_shuffle.iloc[tamanho_treino + tamanho_teste :]

        return x_treino, x_teste, x_valida

    # ------------------------------------------------------------------------
    def inicializa_normalizacao(X_treino: pd.DataFrame) -> MinMaxScaler:
        """
        Função para inicializar MinMaxScaler para normalizar o conjunto de dados com base nos dados de treino

        Args:
            X_treino (pd.DataFrame): O dataset a ser normalizado.

        Returns:
            modelo_norm (MinMaxScaler): O objeto MinMaxScaler ajustado que pode ser usado para normalizar outros conjuntos de dados com base nos dados de treinamento.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        modelo_norm = scaler.fit(X_treino)

        return modelo_norm

    # ------------------------------------------------------------------------
    def normaliza_dados(modelo_normalizador, X) -> pd.DataFrame:
        """
        Função para normalizar os dados usando um modelo de normalização fornecido.

        Args:
            modelo_normalizador: O modelo de normalização a ser usado para normalizar os dados.
            X: Os dados de entrada a serem normalizados.

        Returns:
            X_norm: Os dados normalizados.

        Example Usage:
        ```python

        # Cria um modelo de normalização
        scaler = MinMaxScaler()

        # Normaliza os dados usando o modelo
        normalized_data = normaliza_dados(scaler, X)
        ```
        """

        X_norm = modelo_normalizador.transform(X)
        X_norm_df = pd.DataFrame(X_norm, columns=X.columns)

        return X_norm_df

    # ------------------------------------------------------------------------
    def model_mice(dataset_train):
        imputer = IterativeImputer(max_iter=100)
        mice = imputer.fit(dataset_train.iloc[:, :].values)
        return mice
    
     # ------------------------------------------------------------------------
    def model_dumb(dataset_train):
        imputer = SimpleImputer()
        dumb = imputer.fit(dataset_train.iloc[:, :].values)
        print( type(dumb) )
        print(dumb)
        return dumb

    # ------------------------------------------------------------------------
    def model_knn(dataset_train):
        imputer = KNNImputer(n_neighbors=5)
        knn = imputer.fit(dataset_train.iloc[:, :].values)
        return knn

    # ------------------------------------------------------------------------
    def model_autoencoder_bridge(dataset_train, missing_feature_id, k_perc):
        vae_config = ConfigVAE()
        vae_config.verbose = 0
        vae_config.epochs = 200
        vae_config.neurons = [15]
        vae_config.dropout_rates = [0.1]
        vae_config.latent_dimension = 5
        vae_config.number_features = dataset_train.shape[1]

        vae_bridge_model = VAEBRIDGE(
            vae_config, missing_feature_idx=missing_feature_id, k=k_perc
        )

        vae_bridge_model.fit(dataset_train)

        return vae_bridge_model

    # ------------------------------------------------------------------------
    def model_autoencoder_pmivae(dataset_train):
        original_shape = dataset_train.shape

        vae_config = ConfigVAE()
        vae_config.verbose = 0
        vae_config.epochs = 200
        vae_config.neurons = [15]
        vae_config.dropout_fc = [0.1]
        vae_config.latent_dimension = 5
        vae_config.input_shape = (original_shape[1],)

        pmivae_model = PMIVAE(vae_config, num_samples=200)
        model = pmivae_model.fit(dataset_train)

        return model

    # ------------------------------------------------------------------------
    def modelo_saei(
        dataset_train,
        dataset_test,
        dataset_train_md,
        dataset_test_md,
        nome_coluna,
        input_shape,
    ):
        vae_config = ConfigSAE()
        vae_config.verbose = 0
        vae_config.epochs = 200
        vae_config.input_shape = (input_shape,)

        pmivae_model = SAEImp()

        dados = DataSets(
            x_train=dataset_train,
            x_val=dataset_test,
            x_train_md=dataset_train_md,
            x_val_md=dataset_test_md,
            x_train_pre=dataset_train_md.fillna(np.mean(dataset_train[nome_coluna])),
            x_val_pre=dataset_test_md.fillna(np.mean(dataset_test[nome_coluna])),
        )

        model = pmivae_model.fit(dados, vae_config)
        return model
    
     # ------------------------------------------------------------------------
    
    def model_lpr0(
        datamissing,
        datacomplete,
        target
    ):
        imputer = LabelPropagationRegression0(datacomplete, 
                                              iteracoes=1000, 
                                              epsilon=1e-256)
        
        model = imputer.fit(datamissing, target)
        
        return model


     # ------------------------------------------------------------------------
    
    def model_lpr1(
        datamissing,
        datacomplete,
        target
    ):
        imputer = LabelPropagationRegression1(datacomplete, 
                                              iteracoes=1000, 
                                              epsilon=1e-256)
        
        model = imputer.fit(datamissing, target)
        
        return model
    
     # ------------------------------------------------------------------------
    def model_lpr2(
        datamissing,
        datacomplete,
        target
    ):
        imputer = LabelPropagationRegression2(datacomplete, 
                                              iteracoes=1000, 
                                              epsilon=1e-256)
        
        model = imputer.fit(datamissing, target)
        
        return model
    
    # ------------------------------------------------------------------------
    def model_lpr3(
        datamissing,
        datacomplete,
        target
    ):
        imputer = LabelPropagationRegression3(datacomplete, 
                                              iteracoes=1000, 
                                              epsilon=1e-256)
        
        model = imputer.fit(datamissing, target)
        
        return model

    # ------------------------------------------------------------------------
    def choose_model(model: str, x_train, y_train=None, **kwargs):
        match model:
            case "mice":
                print("[MICE] Training...")
                return MyPipeline.model_mice(x_train)

            case "knn":
                print("[KNN] Training...")
                return MyPipeline.model_knn(x_train)

            case "vaebridge":
                print("[VAEBRIDGE] Training...")
                # Estratégia de pré-imputação, para os VAE é utilizando a média
                X_treino_pre_imput = x_train.fillna(
                    np.mean(x_train[kwargs["col_name"]])
                )

                return MyPipeline.model_autoencoder_bridge(
                    X_treino_pre_imput.loc[:, :].values,
                    kwargs["missing_feature_id"],
                    kwargs["k_perc"],
                )

            case "pmivae":
                print("[PMIVAE] Training...")
                return MyPipeline.model_autoencoder_pmivae(x_train.loc[:, :].values)

            case "saei":
                print("[SAEI] Training...")
                x = kwargs["x_train_md"]
                return MyPipeline.modelo_saei(
                    dataset_train=x_train,
                    dataset_test=kwargs["x_test"],
                    dataset_train_md=kwargs["x_train_md"],
                    dataset_test_md=kwargs["x_test_md"],
                    nome_coluna=kwargs["col_name"],
                    input_shape=kwargs["input_shape"],
                )
            
            case "lpr0":
                print("[LABEL PROPAGATION REGRESSION TIPO 0] Training...")
                return MyPipeline.model_lpr0(
                    datamissing=kwargs["x_train_md"],
                    datacomplete=x_train,
                    target = y_train
                )
            
            case "lpr1":
                print("[LABEL PROPAGATION REGRESSION TIPO 1] Training...")
                return MyPipeline.model_lpr1(
                    datamissing=kwargs["x_train_md"],
                    datacomplete=x_train,
                    target = y_train
                )
            
            case "lpr2":
                print("[LABEL PROPAGATION REGRESSION TIPO 2] Training...")
                return MyPipeline.model_lpr2(
                    datamissing=kwargs["x_train_md"],
                    datacomplete=x_train,
                    target = y_train
                )
            
            case "lpr3":
                print("[LABEL PROPAGATION REGRESSION TIPO 3] Training...")
                return MyPipeline.model_lpr3(
                    datamissing=kwargs["x_train_md"],
                    datacomplete=x_train,
                    target = y_train
                )

            case "mean":
                print("[MEAN] Training...")
                return MyPipeline.model_dumb(x_train)

    # ------------------------------------------------------------------------
    def gera_resultado_univa(
        model_fitted, dataset_normalizado_md, dataset_normalizado_original, missing_id
    ):
        resposta = model_fitted.transform(dataset_normalizado_md.iloc[:, :].values)
        linhas_nan = dataset_normalizado_md.iloc[:, missing_id][
            dataset_normalizado_md.iloc[:, missing_id].isna()
        ].index
        original = dataset_normalizado_original.copy()

        mae = mean_absolute_error(
            y_true=original.iloc[linhas_nan, missing_id],
            y_pred=resposta[linhas_nan, missing_id],
        )

        return mae

    # ------------------------------------------------------------------------
    def gera_resultado_multiva(
        resposta,
        dataset_normalizado_md,
        dataset_normalizado_original,
    ):
        maes = []
        features = dataset_normalizado_md.columns[
            dataset_normalizado_md.isna().any()
        ].tolist()

        for feature in features:
            missing_id = dataset_normalizado_md.columns.get_loc(feature)
            linhas_nan = dataset_normalizado_md.iloc[:, missing_id][
                dataset_normalizado_md.iloc[:, missing_id].isna()
            ].index
            original = dataset_normalizado_original.copy()

            mae = mean_absolute_error(
                y_true=original.iloc[linhas_nan, missing_id],
                y_pred=resposta[linhas_nan, missing_id],
            )
            maes.append(mae)

        return np.mean(maes), np.std(maes), maes

    # ------------------------------------------------------------------------
    def extrai_resultados(tabela_resultados: dict) -> pd.DataFrame:
        fim = []
        for tags in tabela_resultados.keys():
            if tags not in ["datasets", "nome_datasets", "missing_rate"]:
                model_name, nome_dataset, missing_rate, _, error = tags.split("/")

                erro_teste = tabela_resultados[tags]["teste"]

                fim.append((model_name, nome_dataset, missing_rate, error, erro_teste))

        resultados_final = pd.DataFrame(
            fim,
            columns=["Model", "Dataset", "Missing Rate (%)", "Métrica", "Teste"],
        )
        return resultados_final

    # ------------------------------------------------------------------------
    def calcula_metricas_estatisticas_resultados(
        dataset_resultados: pd.DataFrame, nro_metricas: int, nro_iter: int
    ):
        r = []

        for passo in range(0, len(dataset_resultados), nro_metricas * nro_iter):
            filtrada = dataset_resultados[passo : passo + nro_metricas * nro_iter]

            nome_modelo = filtrada["Model"].unique()[0]
            nome = filtrada["Dataset"].unique()[0]
            md = filtrada["Missing Rate (%)"].unique()[0]
            MAE_mean = filtrada["Teste"][filtrada["Métrica"] == "MAE"].mean()
            MAE_std = filtrada["Teste"][filtrada["Métrica"] == "MAE"].std()

            print(
                f"Dataset: {nome} com MD = {md}% MAE = {MAE_mean:.4f} +- {MAE_std:.4f}"
            )

            r.append(
                (
                    nome_modelo,
                    nome,
                    md,
                    round(MAE_mean, 4),
                    round(MAE_std, 4),
                )
            )

        return pd.DataFrame(
            r,
            columns=[
                "Model",
                "Dataset",
                "Missing Rate",
                "MAE_mean",
                "MAE_std",
            ],
        )

    # ------------------------------------------------------------------------
    def carrega_datasets(path_datasets: str) -> dict:
        """
        Carregue conjuntos de dados de um determinado caminho de diretório e retorne-os como um dicionário.

        Argumentos:
            path_datasets (str): O caminho para o diretório que contém os conjuntos de dados.

        Retorna:
            dict: Um dicionário contendo os conjuntos de dados carregados, onde as chaves são os nomes dos arquivos e os valores são DataFrames do pandas.

        Examplo:
            datasets = carrega_datasets('/path/to/datasets')
            print(datasets)
            # Output: {'dataset1': DataFrame1, 'dataset2': DataFrame2, ...}
        """
        datasets_carregados = {}

        for diretorio, subdiretorios, arquivos in os.walk(path_datasets):
            for nome_arquivo in arquivos:
                caminho_completo = os.path.join(diretorio, nome_arquivo)
                nome, extensao = os.path.splitext(nome_arquivo)

                if nome == ".DS_Store" or extensao == ".names":
                    print(".DS_Store or .names ok")
                    continue

                if extensao == ".csv" or extensao == ".data":
                    print("csv or data ok")
                    dados = pd.read_csv(caminho_completo)
                    datasets_carregados[nome] = dados

                elif extensao == ".arff":
                    print(".arff ok")
                    with open(caminho_completo, "r") as f:
                        data = f.read()

                    buffer_texto = StringIO(data)
                    dados, meta = arff.loadarff(buffer_texto)
                    # Convert the numpy array into a dictionary
                    dados_dict = {name: dados[name] for name in dados.dtype.names}

                    # Decode the values to remove the 'b' prefix from the values
                    dados_decodificados = {
                        k: [x.decode() if isinstance(x, bytes) else x for x in v]
                        for k, v in dados_dict.items()
                    }

                    # Convert the decoded data to a pandas DataFrame
                    df = pd.DataFrame(dados_decodificados)
                    datasets_carregados[nome] = df

                elif extensao == ".xls":
                    print(".xls ok")
                    dados = pd.read_excel(
                        caminho_completo, sheet_name="Raw Data", skiprows=1
                    )
                    dados = dados.drop(index=0)

                    datasets_carregados[nome] = dados

                else:
                    raise ValueError(f"Formato de arquivo não encontrado: {extensao}")

        return datasets_carregados

    # ------------------------------------------------------------------------
    def gera_tabela_unificada(tipo: str, *args):
        """
        Generate a unified table by merging data from multiple CSV files.

        Args:
            tipo (str): Type identifier.
            *args: Variable number of positional arguments (file paths).
            **kwargs: Variable number of keyword arguments.

        Returns:
            pd.DataFrame: The final unified table containing the dataset names, missing rates, and mean and standard deviation of MAE for each model.
        """
        dataframes = []
        model_names = []

        for path in args:
            df = pd.read_csv(
                f"./Resultados Parciais {tipo}/{path}.csv", sep=",", index_col=0
            )
            model_name = path.split("_")[0].upper()
            mecanismo = path.split("_")[1]
            dataframes.append(df)
            model_names.append(model_name)

        tabela = {}
        tabela["Dataset"] = dataframes[0]["Dataset"].tolist()
        tabela["Missing Rate"] = dataframes[0]["Missing Rate"].tolist()

        for df, model_name in zip(dataframes, model_names):
            tabela[model_name] = [
                f"{mean} ± {std}" for mean, std in zip(df["MAE_mean"], df["MAE_std"])
            ]

        final = pd.DataFrame(tabela)

        print(f"Resultados salvos com sucesso!")
        final.to_csv(f"Resultados {tipo}/{mecanismo.upper()}.csv", sep=",")
        return final

    # ------------------------------------------------------------------------
    def analisa_complexidade(path):
        complexity = Complexity(path)
        return {
            "f1": min(complexity.F1()),
            "l1_mean": np.mean(complexity.l1()),
            "l1_std": np.std(complexity.l1()),
            "n1": complexity.N1(),
            "n2": complexity.N2(),
            "n3": complexity.N3(),
            #"t1": complexity.T1(),
        }

    # ------------------------------------------------------------------------
    def heatmap_resultados(path: str, mecanismo: str, estrategia: str):
        all_heatmaps = {}

        print("PATH: ", path)
        resultados = pd.read_csv(path, index_col=0)
        print(resultados)

        for nome in resultados.Dataset.unique():
            
            dataset = resultados[resultados.Dataset == nome].reset_index(drop=True)

            dataset = dataset.copy()

            for i in dataset.columns[2:]:
                dataset[i] = dataset[i].str.extract(r"([0-9.]+) ±")

            colunas = {0: "MR = 10%", 1: "MR = 20%", 2: "MR = 40%", 3: "MR = 60%"}

            df = dataset.iloc[:, 2:].T.rename(columns=colunas)
            styled_df = df.style.background_gradient(
                cmap="Grays", subset=["MR = 10%", "MR = 20%", "MR = 40%", "MR = 60%"]
            )
            all_heatmaps[nome] = styled_df

        output = f"Análises Resultados/{mecanismo}-{estrategia}.xlsx"
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for key in all_heatmaps.keys():
                all_heatmaps[key].to_excel(writer, sheet_name=key)





# A partir daqui:

import os
import pandas as pd
from scipy.io import arff
from io import StringIO

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Label Propagation for Missing Data Imputation
from LPMD import *

# Variational Autoencoder Filter for Bayesian Ridge Imputation
from Autoencoders.bridge import VAEBRIDGE
from Autoencoders.vae_bridge import ConfigVAE

# MICE, KNN, Dumb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

# Partial Multiple Imputation with Variational Autoencoders
from Autoencoders.pmivae import PMIVAE
from Autoencoders.vae_pmivae import ConfigVAE

# Siamese Autoencoder
from Autoencoders.saei import ConfigSAE, SAEImp, DataSets

# Libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from pycol import Complexity
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# ==========================================================================
class MyPipeline:
    # ------------------------------------------------------------------------
    def label_encoder(df, lista_nome_colunas):
        """
        Encodes categorical columns using LabelEncoder.

        Args:
            df: DataFrame to be processed.
            lista_nome_colunas: List of column names to encode.

        Returns:
            DataFrame with encoded columns.
        """
        data = df.copy()
        le = LabelEncoder()

        for att in lista_nome_colunas:
            data[att] = le.fit_transform(data[att])

        return data

    # ------------------------------------------------------------------------
    def one_hot_encode(df, lista_nome_colunas):
        """
        Performs one-hot encoding on specified columns.

        Args:
            df: DataFrame to be processed.
            lista_nome_colunas: List of column names to encode.

        Returns:
            DataFrame with one-hot encoded columns.
        """
        data = df.copy()
        data_categorical = data[lista_nome_colunas]

        encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = encoder.fit_transform(data_categorical)

        # Create a new DataFrame with the one-hot encoded columns
        columns = encoder.get_feature_names_out(lista_nome_colunas)
        df_encoded = pd.DataFrame(one_hot_encoded, columns=columns)

        # Concatenate the original DataFrame with the encoded DataFrame
        data = pd.concat([data, df_encoded], axis=1)

        # Drop the original columns
        data = data.drop(columns=lista_nome_colunas)

        return data

    # ------------------------------------------------------------------------
    def cria_tabela_resultados(datasets):
        """
        Creates a results table with datasets and their corresponding metadata.

        Args:
            datasets: Dictionary of datasets.

        Returns:
            A dictionary containing the results table.
        """
        tabela_resultados = {}

        (
            breast_cancer_wisconsin_df,
            pima_diabetes_df,
            bc_coimbra_df,
            indian_liver_df,
            parkinsons_df,
            mammographic_masses_df,
            hcv_egyptian_df,
            thoracic_surgery_df,
        ) = MyPipeline.pre_processing_datasets(datasets)

        tabela_resultados["datasets"] = [
            breast_cancer_wisconsin_df,
            pima_diabetes_df,
            bc_coimbra_df,
            indian_liver_df,
            parkinsons_df,
            mammographic_masses_df,
            hcv_egyptian_df,
            thoracic_surgery_df,
        ]
        tabela_resultados["nome_datasets"] = [
            "wiscosin",
            "pima_diabetes",
            "bc_coimbra",
            "indian_liver",
            "parkinsons",
            "mammographic_masses",
            "hcv_egyptian",
            "thoracic_surgey",
        ]
        tabela_resultados["missing_rate"] = [10, 20, 40, 60]

        return tabela_resultados

    # ------------------------------------------------------------------------
    def pre_processing_datasets(datasets: dict):
        """
        Preprocesses datasets by cleaning, encoding, and preparing them.

        Args:
            datasets: Dictionary containing raw datasets.

        Returns:
            A tuple of processed datasets.
        """
        breast_cancer_wisconsin_df = datasets["wiscosin"].copy()
        breast_cancer_wisconsin_df = breast_cancer_wisconsin_df.drop(columns="ID")
        breast_cancer_wisconsin_df = MyPipeline.label_encoder(
            breast_cancer_wisconsin_df, ["target"]
        )

        pima_diabetes_df = datasets["pima_diabetes"].copy()

        bc_coimbra_df = datasets["bc_coimbra"].copy()

        indian_liver_df = datasets["indian_liver"].copy()
        indian_liver_df = indian_liver_df.dropna()
        indian_liver_df = MyPipeline.label_encoder(indian_liver_df, ["Gender"])

        datasets["parkinsons"] = datasets["parkinsons"].drop(columns="name")
        parkinsons_df = datasets["parkinsons"].copy()

        mammographic_masses_df = datasets["mammographic_masses"].copy()
        mammographic_masses_df = (
            datasets["mammographic_masses"]
            .replace("?", np.nan)
            .dropna()
            .drop(columns="BI-RADS assessment")
            .reset_index(drop=True)
        )
        mammographic_masses_df["Age"] = mammographic_masses_df["Age"].astype("int64")
        mammographic_masses_df["Density"] = mammographic_masses_df["Density"].astype(
            "int64"
        )
        mammographic_masses_df = MyPipeline.one_hot_encode(
            mammographic_masses_df, ["Shape", "Margin"]
        )

        hcv_egyptian_df = datasets["HCV-Egy-Data"].copy()

        thoracic_surgery_df = datasets["ThoraricSurgery"].copy()
        thoracic_surgery_df = MyPipeline.label_encoder(
            MyPipeline.one_hot_encode(thoracic_surgery_df, ["DGN"]),
            [
                "PRE6",
                "PRE7",
                "PRE8",
                "PRE9",
                "PRE10",
                "PRE11",
                "PRE14",
                "PRE17",
                "PRE19",
                "PRE25",
                "PRE30",
                "PRE32",
                "target",
            ],
        )

        return (
            breast_cancer_wisconsin_df,
            pima_diabetes_df,
            bc_coimbra_df,
            indian_liver_df,
            parkinsons_df,
            mammographic_masses_df,
            hcv_egyptian_df,
            thoracic_surgery_df,
        )

    # ------------------------------------------------------------------------
    def cria_dataframe(df) -> pd.DataFrame:
        """
        Creates a pandas DataFrame from sklearn datasets.

        Args:
            df: A pandas DataFrame.

        Returns:
            A pandas DataFrame with an additional 'target' column.
        """
        dataset = pd.DataFrame(data=df.data, columns=df.feature_names)
        dataset["target"] = df.target
        return dataset

    # ------------------------------------------------------------------------
    def split_dataset(dataset: pd.DataFrame, perc_treino: float, perc_teste: float):
        """
        Splits the dataset into training, testing, and validation sets.

        Args:
            dataset (pd.DataFrame): DataFrame to split.
            perc_treino (float): Training set percentage.
            perc_teste (float): Testing set percentage.

        Returns:
            tuple: (X_treino, X_teste, X_valida)
        """
        dataset = dataset.copy()
        df_shuffle = dataset.sample(frac=1.0, replace=True)

        tamanho_treino = int(perc_treino * len(dataset))
        tamanho_teste = int(perc_teste * len(dataset))

        x_treino = df_shuffle.iloc[:tamanho_treino]
        x_teste = df_shuffle.iloc[tamanho_treino : tamanho_treino + tamanho_teste]
        x_valida = df_shuffle.iloc[tamanho_treino + tamanho_teste :]

        return x_treino, x_teste, x_valida

    # ------------------------------------------------------------------------
    def inicializa_normalizacao(X_treino: pd.DataFrame) -> MinMaxScaler:
        """
        Initializes MinMaxScaler to normalize training data.

        Args:
            X_treino (pd.DataFrame): Training dataset.

        Returns:
            MinMaxScaler: Fitted scaler for normalization.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        modelo_norm = scaler.fit(X_treino)

        return modelo_norm

    # ------------------------------------------------------------------------
    def normaliza_dados(modelo_normalizador, X) -> pd.DataFrame:
        """
        Normalizes data using a provided normalization model.

        Args:
            modelo_normalizador: Normalization model.
            X: Input data to normalize.

        Returns:
            Normalized DataFrame.
        """
        X_norm = modelo_normalizador.transform(X)
        X_norm_df = pd.DataFrame(X_norm, columns=X.columns)

        return X_norm_df

    # ------------------------------------------------------------------------
    def model_mice(dataset_train):
        """
        Builds a MICE imputation model.

        Args:
            dataset_train: Training dataset.

        Returns:
            Fitted MICE model.
        """
        imputer = IterativeImputer(max_iter=100)
        mice = imputer.fit(dataset_train.iloc[:, :].values)
        return mice
    
    # ------------------------------------------------------------------------
    def model_dumb(dataset_train):
        """
        Builds a simple imputation model.

        Args:
            dataset_train: Training dataset.

        Returns:
            Fitted simple imputation model.
        """
        imputer = SimpleImputer()
        dumb = imputer.fit(dataset_train.iloc[:, :].values)
        return dumb

    # ------------------------------------------------------------------------
    def model_knn(dataset_train):
        """
        Builds a KNN imputation model.

        Args:
            dataset_train: Training dataset.

        Returns:
            Fitted KNN model.
        """
        imputer = KNNImputer(n_neighbors=5)
        knn = imputer.fit(dataset_train.iloc[:, :].values)
        return knn

    # ------------------------------------------------------------------------
    def model_autoencoder_bridge(dataset_train, missing_feature_id, k_perc):
        """
        Builds a VAE Bridge model for imputation.

        Args:
            dataset_train: Training dataset.
            missing_feature_id: Index of the missing feature.
            k_perc: Percentage parameter for the model.

        Returns:
            Trained VAE Bridge model.
        """
        vae_config = ConfigVAE()
        vae_config.verbose = 0
        vae_config.epochs = 200
        vae_config.neurons = [15]
        vae_config.dropout_rates = [0.1]
        vae_config.latent_dimension = 5
        vae_config.number_features = dataset_train.shape[1]

        vae_bridge_model = VAEBRIDGE(
            vae_config, missing_feature_idx=missing_feature_id, k=k_perc
        )

        vae_bridge_model.fit(dataset_train)

        return vae_bridge_model

    # ------------------------------------------------------------------------
    def model_autoencoder_pmivae(dataset_train):
        """
        Builds a PMIVAE model for imputation.

        Args:
            dataset_train: Training dataset.

        Returns:
            Trained PMIVAE model.
        """
        original_shape = dataset_train.shape

        vae_config = ConfigVAE()
        vae_config.verbose = 0
        vae_config.epochs = 200
        vae_config.neurons = [15]
        vae_config.dropout_fc = [0.1]
        vae_config.latent_dimension = 5
        vae_config.input_shape = (original_shape[1],)

        pmivae_model = PMIVAE(vae_config, num_samples=200)
        model = pmivae_model.fit(dataset_train)

        return model

    # ------------------------------------------------------------------------
    def modelo_saei(
        dataset_train,
        dataset_test,
        dataset_train_md,
        dataset_test_md,
        nome_coluna,
        input_shape,
    ):
        """
        Builds a SAEI model for imputation.

        Args:
            dataset_train: Training dataset.
            dataset_test: Testing dataset.
            dataset_train_md: Missing data in training set.
            dataset_test_md: Missing data in testing set.
            nome_coluna: Column name with missing values.
            input_shape: Input shape for the model.

        Returns:
            Trained SAEI model.
        """
        vae_config = ConfigSAE()
        vae_config.verbose = 0
        vae_config.epochs = 200
        vae_config.input_shape = (input_shape,)

        pmivae_model = SAEImp()

        dados = DataSets(
            x_train=dataset_train,
            x_val=dataset_test,
            x_train_md=dataset_train_md,
            x_val_md=dataset_test_md,
            x_train_pre=dataset_train_md.fillna(np.mean(dataset_train[nome_coluna])),
            x_val_pre=dataset_test_md.fillna(np.mean(dataset_test[nome_coluna])),
        )

        model = pmivae_model.fit(dados, vae_config)
        return model

    # ------------------------------------------------------------------------



