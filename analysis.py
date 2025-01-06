from MyUtils import MyPipeline

path = "/home/filipe/Documentos/GitHub/LPMD/finalResults/MNAR-MAR.csv"

mecanismo = "mnar"
estrategia = "mar_median"

MyPipeline.heatmap_resultados(path, mecanismo, estrategia)
     