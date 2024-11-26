import pandas as pd

def diff(nome, df):
    F1 = base[nome][0] - df['f1_mean']
    L1 = base[nome][1] - df['l1_mean']
    N1 = base[nome][3] - df['n1_mean']
    N2 = base[nome][4] - df['n2_mean']
    N3 = base[nome][5] - df['n3_mean']

    return F1.to_list(), L1.to_list(), N1.to_list(), N2.to_list(), N3.to_list()

def tabela_complexidade_diferenca(dataframe):

    results = {}

    results["imputer"] = dataframe["imputer"].to_list()
    results["Dataset"] = dataframe["dataset"].to_list()
    results["Missing Rate"] = dataframe["missing rate"].to_list()

    results.update({"F1": [],
        "L1": [],
        "N1": [],
        "N2": [],
        "N3": []})

    datasets = ["wiscosin", "pima_diabetes", "bc_coimbra", "indian_liver", "parkinsons", "mammographic_masses", "hcv_egyptian", "thoracic_surgey"]

    passo = 4

    for dataset in datasets:
        F1, L1, N1, N2, N3 = diff(dataset, dataframe[datasets.index(dataset) * passo : datasets.index(dataset) * passo + passo])

        for VF1, VL1, VN1, VN2, VN3 in zip(F1, L1, N1, N2, N3):
            results["F1"].append(VF1)
            results["L1"].append(VL1)
            results["N1"].append(VN1)
            results["N2"].append(VN2)
            results["N3"].append(VN3)

    return pd.DataFrame(results)

if __name__ == "__main__":

    path_all = "C:\\Users\\Mult-e\\Desktop\\@Codigos\\MestradoCodigos\\MestradoCodigos\\all_complexity.xlsx"
    path_baseline = "C:\\Users\\Mult-e\\Desktop\\@Codigos\\MestradoCodigos\\MestradoCodigos\\Análises Resultados\\Complexidade\\baseline.xlsx"

    base = pd.read_excel(path_baseline)
    all_cmp = pd.read_excel(path_all)

    # Cada imputer tem um dataset próprio
    mean = all_cmp[:32]
    knn = all_cmp[32:64]
    mice = all_cmp[64:96]
    pmivae = all_cmp[96:128]
    saei = all_cmp[128:160]

    mean_dif = tabela_complexidade_diferenca(mean)
    knn_dif = tabela_complexidade_diferenca(knn)
    mice_dif = tabela_complexidade_diferenca(mice)
    pmivae_dif = tabela_complexidade_diferenca(pmivae)
    saei_dif = tabela_complexidade_diferenca(saei)

    all_imputers = pd.concat([mean_dif, knn_dif, mice_dif, pmivae_dif, saei_dif])
    all_imputers.to_excel("C:\\Users\\Mult-e\\Desktop\\@Codigos\\MestradoCodigos\\MestradoCodigos\\Análises Resultados\\Complexidade\\Diferença complexidade com baseline - mnar_mbouv.xlsx",
                        index=False)