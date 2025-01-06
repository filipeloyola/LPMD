from MyUtils import MyPipeline

mecanismo = "mnar"
tipo = "Multivariado"

for i in ["mar_median"]: # exactly as placed in the mechanism of the file "multivariateExperimentsClassification.py"
    path_knn = f"knn_{mecanismo}-{i}"
    path_mice = f"mice_{mecanismo}-{i}"
    path_vae = f"vaebridge_{mecanismo}-{i}"
    path_pmivae = f"pmivae_{mecanismo}-{i}"
    path_saei = f"saei_{mecanismo}-{i}"
    path_dumb = f"mean_{mecanismo}-{i}"
    path_lpmd = f"lpmd_{mecanismo}-{i}"
    path_lpmd2 = f"lpmd2_{mecanismo}-{i}"

    MyPipeline.gera_tabela_unificada(tipo,
        path_knn, path_mice, path_pmivae, path_saei, path_dumb, path_lpmd, path_lpmd2
    )

