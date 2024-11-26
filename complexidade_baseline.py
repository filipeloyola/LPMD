import arff
from MyUtils import MyPipeline
import pandas as pd

def cria_arquivo_arff(tabela_resultados):
    for dados, nome in zip(tabela_resultados["datasets"], tabela_resultados["nome_datasets"]):
        
        target_values = dados.target
        dados = dados.drop(columns = "target")
        dados["target"] = target_values
        attributes = []

        for j in dados:
            if dados[j].dtypes in ['int64', 'float64', 'float32'] and j != "target":
                attributes.append((j, 'NUMERIC'))
            elif j == "target":
                if nome == "bc_coimbra" or nome == "indian_liver":
                    attributes.append((j, ['1.0','2.0']))
                elif nome == "hcv_egyptian":
                    attributes.append((j, dados[j].unique().astype(str).tolist()))
                else:
                    attributes.append((j, ['1.0','0.0']))
            else:
                attributes.append((j, dados[j].unique().astype(str).tolist()))


        dictarff = {'attributes': attributes,
                    'data': dados.values.tolist(),
                    'relation': f"{nome}"}

        # Criar o arquivo ARFF
        arff_content = arff.dumps(dictarff)
        
        # Salvar o conteúdo ARFF em um arquivo
        with open(f"./Análises Resultados/Complexidade/baseline/{nome}.arff", "w") as fcom:
            fcom.write(arff_content)

if __name__ == "__main__":

    diretorio_principal = "C:\\Users\\Mult-e\\Desktop\\@Codigos\\Datasets"
    datasets = MyPipeline.carrega_datasets(diretorio_principal)

    tabela_resultados = MyPipeline.cria_tabela_resultados(datasets)
    # cria_arquivo_arff(tabela_resultados)

    bs = {}
    for nome in tabela_resultados["nome_datasets"]:
        path = f"Análises Resultados\\Complexidade\\baseline\\{nome}.arff"
        bs[nome]=MyPipeline.analisa_complexidade(path)

    pd.DataFrame(bs).to_excel("Análises Resultados\\Complexidade\\baseline.xlsx")