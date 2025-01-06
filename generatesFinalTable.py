from MyUtils import MyPipeline

mecanismo = "mnar"
tipo = "Multivariado"

for i in ["mar_median"]: # exatamente como colocado no mecanismo do arquivo "experimentos_multivariado_classifica.py"
    path_knn = f"knn_{mecanismo}-{i}"
    path_mice = f"mice_{mecanismo}-{i}"
    path_vae = f"vaebridge_{mecanismo}-{i}"
    path_pmivae = f"pmivae_{mecanismo}-{i}"
    path_saei = f"saei_{mecanismo}-{i}"
    path_dumb = f"mean_{mecanismo}-{i}"
    path_lpr0 = f"lpr0_{mecanismo}-{i}"
    path_lpr1 = f"lpr1_{mecanismo}-{i}"
    path_lpr2 = f"lpr2_{mecanismo}-{i}"
    path_lpr3 = f"lpr3_{mecanismo}-{i}"

    MyPipeline.gera_tabela_unificada(tipo,
        path_knn, path_mice, path_pmivae, path_saei, path_dumb, path_lpr0, path_lpr1, path_lpr2, path_lpr3
    )

