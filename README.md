# LPMD: Label Propagation for Missing Data Imputation

This repository contains files about the article "A Label Propagation Approach for Missing Data Imputation" submitted to the IEEE Access jornal. In this repository, you can find the ".py" files used in the research, as well as directories containing the datastes and partial results. The LPMD algorithm is available in the file "LPMD.py". To use this pipeline, follow these steps:

- Rodar o código presente no ```experimentos_multivariado_classifica.py``` variando os imputadores entre: mean, knn, mice, pmivae, saei; <br>

Com esta etapa cumprida, os resultados da classificação usando a Árvore de Decisão, os arquivos.arff para a análise de complexidade, os resultados da imputação e os tempos de treinamento de cada algoritmo já foram gerados. Portanto, basta seguir os próximos passos:

- Rodar o código ```gera_tabela_final.py``` para ter os resultados de todos os imputadores em um mesmo arquivo Excel;
- Rodar o código ```analises.py``` para ter um heatmap dos MAE de cada imputador;
- No Excel, formatar a tabela de resultados para a métrica MAE;

- Para a classificação, rodar o código ```gera_tabela_classificacao.py``` para ter uma sheet os resultados da acurácia, outra dee precisão e por fim recall da Árvore de Decisão
- No excel, formatar a tabela e colocar uma formatação condicional para os valores maiores que zero;

- Para a complexidade, rodar o código ```complexity.py``` que irá gerar 5 arquivos Excel contendo, em cada sheet, as métricas de complexidade para cada fold de cada missing rate em cada dataset.
- Rodar o código ```complexity_all.py``` para unificar todas as métricas de complexidade em um único arquivo Excel
- Rodar o código ```complexidade_baseline.py``` para calcular as métricas de complexidade no baseline
- Rodar o código ```calcula_dif_complexidade.py``` para calcular a diferença entre o baseline e as médias de cada fold para métricas as métricas de complexidade;
- No excel, formatar a tabela e colocar uma formatação condicional para os valores maiores que zero;
