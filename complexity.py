from MyUtils import MyPipeline
import pandas as pd
from time import perf_counter

fold = 0
list_dataset = ["wiscosin",
            "pima_diabetes",
            "bc_coimbra",
            "indian_liver",
            "parkinsons",
            "mammographic_masses",
            "hcv_egyptian",
            "thoracic_surgey"]

for model_impt in ["mean", "mice", "pmivae", "saei"]:
    results = {}

    inicio = perf_counter()
    output = f"Análises Resultados/Complexidade/{model_impt}_complexity.xlsx"

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:

        for dataset in list_dataset:
            for md in [10,20,40,60]:
                for fold in range(5):
                    print(f"Dataset: {dataset} com md = {md} no fold = {fold}")
                    path = f"./Análises Resultados/Complexidade/{model_impt}/{model_impt}_fold{fold}_{dataset}_{md}.arff"
                    dict_complexity = MyPipeline.analisa_complexidade(path)
                    results[f"fold{fold}"] = dict_complexity

                
                df_results = pd.DataFrame(results)
                df_results.to_excel(writer, sheet_name=f'{dataset}_{md}')

    fim = perf_counter()
    print(f"Tempo de duração para {model_impt}: {fim-inicio:.4f}")
        