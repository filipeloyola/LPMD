import pandas as pd 
import numpy as np 
from openpyxl import load_workbook
from MyUtils import MyPipeline

def tabela_complexidade(imputer:str):

    wb = load_workbook(f"Análises Resultados/Complexidade/{imputer}_complexity.xlsx")

    results = {"imputer":[],
            "dataset":[],
            "f1_mean":[],
            "f1_std":[],
            "l1_mean":[],
            "l1_std":[],
            "n1_mean":[],
            "n1_std":[],
            "n2_mean":[],
            "n2_std":[],
            "n3_mean":[],
            "n3_std":[],
            }

    for sheet in wb:
        dataset = sheet.title
        results["dataset"].append(dataset)
        results["imputer"].append(imputer)

        # Lista para armazenar os valores das células
        valores_f1 = []
        valores_l1 = []
        valores_n1 = []
        valores_n2 = []
        valores_n3 = []

        # Colunas de interesse
        colunas_interesse = ['B', 'C', 'D', 'E', 'F']

        # Iterar sobre as colunas
        for coluna in colunas_interesse:
            # Construir a coordenada da célula (ex: B2, C2, ...)
            f1 = f'{coluna}2'
            l1 = f'{coluna}3'
            n1 = f'{coluna}5'
            n2 = f'{coluna}6'
            n3 = f'{coluna}7'

            valor_f1 = sheet[f1].value
            valores_f1.append(valor_f1)
            valor_l1 = sheet[l1].value
            valores_l1.append(valor_l1)
            valor_n1 = sheet[n1].value
            valores_n1.append(valor_n1)
            valor_n2 = sheet[n2].value
            valores_n2.append(valor_n2)
            valor_n3 = sheet[n3].value
            valores_n3.append(valor_n3)

    

        f1_mean, f1_std = np.mean(valores_f1), np.std(valores_f1)
        l1_mean, l1_std = np.mean(valores_l1), np.std(valores_l1)
        n1_mean, n1_std = np.mean(valores_n1), np.std(valores_n1)
        n2_mean, n2_std = np.mean(valores_n2), np.std(valores_n2)
        n3_mean, n3_std = np.mean(valores_n3), np.std(valores_n3)

        results["f1_mean"].append(f1_mean)
        results["f1_std"].append(f1_std)

        results["l1_mean"].append(l1_mean)
        results["l1_std"].append(l1_std)

        results["n1_mean"].append(n1_mean)
        results["n1_std"].append(n1_std)

        results["n2_mean"].append(n2_mean)
        results["n2_std"].append(n2_std)

        results["n3_mean"].append(n3_mean)
        results["n3_std"].append(n3_std)

    return pd.DataFrame(results)

if __name__ == "__main__":
    knn = tabela_complexidade("knn")
    mean = tabela_complexidade("mean")
    mice = tabela_complexidade("mice")
    pmivae = tabela_complexidade("pmivae")
    saei = tabela_complexidade("saei")

    mecanismo = "MNAR-MBOUV"

    # Todos os resultados de complexidade em um mesmo Excel
    all_results = pd.concat([mean, knn, mice, pmivae, saei])
    all_results.to_excel(f"{mecanismo}_all_complexity.xlsx", index=False)

    print("Resultados salvos com sucesso!")