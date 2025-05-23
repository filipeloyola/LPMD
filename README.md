# LPMD: Label Propagation for Missing Data Imputation

This repository contains files about the article "A Label Propagation Approach for Missing Data Imputation" submitted to the IEEE Access journal. In this repository, you can find the ".py" files used in the research, as well as directories containing the datasets and partial results. The LPMD algorithm is available in the file  ```LPMD.py```. To use this pipeline, follow these steps:

- Run the code ```multivariateExperimentsClassification.py``` varying the inputs between: mean, knn, mice, pmivae, saei, lpmd

- Run the code ```generatesFinalTable.py``` to have the results of all the inputs in the same spreadsheet file;
  
- Run the code ```analysis.py``` to have a heatmap of the MAE of each input;
  
- In Excel, format the results table for the MAE metric;

- For classification, run the code ```generatesClassificationTable.py``` to have a spreadsheet with the accuracy results, another with precision and finally recall;
  
- In Excel, format the table and place conditional formatting for values ​​greater than zero;

## Research Article

https://doi.org/10.1109/ACCESS.2025.3559772

## Citation Research Article
If you use LPMD or this article in your research, please cite as below.

@article{LOPES2025,
  title={A Label Propagation Approach for Missing Data Imputation},
  journal={IEEE Access},
  volume={13},
  pages={65925-65938},
  year={2025},
  issn={2169-3536},
  doi={[10.1109/ACCESS.2025.3559772](https://doi.org/10.1109/ACCESS.2025.3559772)},
  author={Loyola Lopes, Filipe and Dantas Mangussi, Arthur and Cardoso Pereira, Ricardo and Seoane Santos, Miriam and Henriques Abreu, Pedro and Carolina Lorena, Ana}
  }


## Citation for the mdatagen package by Mangussi et al. (2025):
If you need to use the data amputation library, please also cite the research below.

@article{MANGUSSI2025,
title = {mdatagen: A python library for the artificial generation of missing data},
journal = {Neurocomputing},
volume = {625},
pages = {129478},
year = {2025},
issn = {0925-2312},
doi = {[10.1016/j.neucom.2025.129478](https://doi.org/10.1016/j.neucom.2025.129478)},
author = {Arthur Dantas Mangussi and Miriam Seoane Santos and Filipe Loyola Lopes and Ricardo Cardoso Pereira and Ana Carolina Lorena and Pedro Henriques Abreu}
}

## Acknowledgements
This work was supported in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code
001 and Brazilian funding agencies FAPESP (Fundação Amparo à Pesquisa do Estado de São Paulo) under grants 2022/10553-6,
2023/13688-2, and 2021/06870-3.

