# LPMD: Label Propagation for Missing Data Imputation

This repository contains files about the article "A Label Propagation Approach for Missing Data Imputation" submitted to the IEEE Access journal. In this repository, you can find the ".py" files used in the research, as well as directories containing the datasets and partial results. The LPMD algorithm is available in the file  ```LPMD.py```. To use this pipeline, follow these steps:

- Run the code in ```experimentos_multivariado_classifica.py``` varying the inputs between: mean, knn, mice, pmivae, saei, lpmd

- Run the code ```gera_tabela_final.py``` to have the results of all the inputs in the same spreadsheet file;
  
- Run the code ```analises.py``` to have a heatmap of the MAE of each input;
  
- In Excel, format the results table for the MAE metric;

- For classification, run the code ```gera_tabela_classificacao.py``` to have a spreadsheet with the accuracy results, another with precision and finally recall;
  
- In Excel, format the table and place conditional formatting for values ​​greater than zero;

## Citation
In construction.

## Acknowledgements
This research was supported in part by the Coordenação de Aperfeiçoamento de Pessoalde Nível Superior - Brasil (CAPES) - Finance Code 001. The authors gratefully acknowledge the Brazilian funding agency FAPESP (Fundação Amparo à Pesquisa do Estado de São Paulo) for the COVID-19 Data Sharing repository.

