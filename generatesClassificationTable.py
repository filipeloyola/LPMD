import pandas as pd

def calcula_dif(df):
    data = df.copy()
    for cont in range(len(data.dataset.unique())):
        data.loc[data.dataset == df_base.loc[cont, 'dataset'], 'accuracy'] -= df_base.loc[cont, 'accuracy']
        data.loc[data.dataset == df_base.loc[cont, 'dataset'], 'recall'] -= df_base.loc[cont,'recall']
        data.loc[data.dataset == df_base.loc[cont, 'dataset'], 'precision'] -= df_base.loc[cont,'precision']

    return data

def rename_col(df, nome):
    return df.rename(columns={"accuracy": f"{nome}_accuracy",
                       "recall": f"{nome}_recall",
                       "precision": f"{nome}_precision"})

#path = "C:\\Users\\Mult-e\\Desktop\\@Codigos\\MestradoCodigos\\MestradoCodigos\\Análises Resultados\\Classificação\\"
path = "/home/filipe/Documentos/GitHub/LabelPropagationComparison/Análises Resultados/Classificação/"


df_base = pd.read_csv(f"{path}RandomForest_baseline.csv", index_col=0) # baseline
df_knn = pd.read_csv(f"{path}knn_DecisionTree.csv", index_col=0)
df_mean = pd.read_csv(f"{path}mean_DecisionTree.csv", index_col=0)
df_mice = pd.read_csv(f"{path}mice_DecisionTree.csv", index_col=0)
df_pmivae = pd.read_csv(f"{path}pmivae_DecisionTree.csv", index_col=0)
df_saei = pd.read_csv(f"{path}saei_DecisionTree.csv", index_col=0)
df_lpr0 = pd.read_csv(f"{path}lpr0_DecisionTree.csv", index_col=0)
df_lpr1 = pd.read_csv(f"{path}lpr1_DecisionTree.csv", index_col=0)
df_lpr2 = pd.read_csv(f"{path}lpr2_DecisionTree.csv", index_col=0)
df_lpr3 = pd.read_csv(f"{path}lpr3_DecisionTree.csv", index_col=0)

df_knn = calcula_dif(df_knn)
df_mean = calcula_dif(df_mean)
df_mice = calcula_dif(df_mice)
df_pmivae = calcula_dif(df_pmivae)
df_saei = calcula_dif(df_saei)
df_lpr0 = calcula_dif(df_lpr0)
df_lpr1 = calcula_dif(df_lpr1)
df_lpr2 = calcula_dif(df_lpr2)
df_lpr3 = calcula_dif(df_lpr3)


df_knn = rename_col(df_knn, "knn")
df_mean = rename_col(df_mean, "mean")
df_mice = rename_col(df_mice, "mice")
df_pmivae = rename_col(df_pmivae, "pmivae")
df_saei = rename_col(df_saei, "saei")
df_lpr0 = rename_col(df_lpr0, "lpr0")
df_lpr1 = rename_col(df_lpr1, "lpr1")
df_lpr2 = rename_col(df_lpr2, "lpr2")
df_lpr3 = rename_col(df_lpr3, "lpr3")


df1 = df_knn[["dataset", "missing_rate", "knn_accuracy"]].join(df_mice["mice_accuracy"])
df1 = df1.join(df_saei["saei_accuracy"])
df1 = df1.join(df_pmivae["pmivae_accuracy"])
df1 = df1.join(df_mean["mean_accuracy"])
df1 = df1.join(df_lpr0["lpr0_accuracy"])
df1 = df1.join(df_lpr1["lpr1_accuracy"])
df1 = df1.join(df_lpr2["lpr2_accuracy"])
df1 = df1.join(df_lpr3["lpr3_accuracy"])

df2 = df_knn[["dataset", "missing_rate", "knn_recall"]].join(df_mice["mice_recall"])
df2 = df2.join(df_saei["saei_recall"])
df2 = df2.join(df_pmivae["pmivae_recall"])
df2 = df2.join(df_mean["mean_recall"])
df2 = df2.join(df_lpr0["lpr0_recall"])
df2 = df2.join(df_lpr1["lpr1_recall"])
df2 = df2.join(df_lpr2["lpr2_recall"])
df2 = df2.join(df_lpr3["lpr3_recall"])

df3 = df_knn[["dataset", "missing_rate", "knn_precision"]].join(df_mice["mice_precision"])
df3 = df3.join(df_saei["saei_precision"])
df3 = df3.join(df_pmivae["pmivae_precision"])
df3 = df3.join(df_mean["mean_precision"])
df3 = df3.join(df_lpr0["lpr0_precision"])
df3 = df3.join(df_lpr1["lpr1_precision"])
df3 = df3.join(df_lpr2["lpr2_precision"])
df3 = df3.join(df_lpr3["lpr3_precision"])


output = f"{path}Diferença dos algoritmos para baseline.xlsx"
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    df1.to_excel(writer, sheet_name = "Accuracy", index=False)
    df2.to_excel(writer, sheet_name = "Recall", index=False)
    df3.to_excel(writer, sheet_name = "Precision", index=False)

print("Resultados salvos com sucesso")