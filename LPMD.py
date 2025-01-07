#########################################################
# modules
#########################################################

import sklearn as sk
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.impute import SimpleImputer
#from sklearn.cluster import AgglomerativeClustering

# MICE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# K-means
from sklearn.cluster import KMeans

#########################################################
# functions
#########################################################

def compara_original(A, B):
    ''' Replaces the values of A with the values of B where B is not np.nan.
        Args:
        A: A np.array representing the current Y.
        B: A np.array with the same dimensions as A, representing the reference initial data.
        Returns:
        A np.array with the replaced values.
    '''
    A[~np.isnan(B)] = B[~np.isnan(B)]
    return A

# Main algorithm function
def label_propagation(data_missing, data_mice, epsilon, max_iter):
    ''' data_missing: np.array containing missing values.
        data_mice: np.array with imputation by the chosen method.
        return: np.array filled by Label Propagation.
        note: This function must be called for one cluster at a time.
        epsilon: Argument for the algorithm.
        max_iter: Maximum number of iterations.
    '''
    
    # Defining dataref
    dataref = data_missing.copy()

    # Defining matrix Y
    Y = data_mice.copy()

    # Creating matrix W
    W = rbf_kernel(Y) # RBF kernel from sklearn with gamma=20
   
    # Creating transition matrix T 
    normalizer = W.sum(axis=0)     # Initializes the normalizer variable with the sum of each column
    W /= normalizer[:, np.newaxis] # Each value of W is divided by the normalizer value for each column
    T = W

    Y_prev = np.zeros((Y.shape[0], Y.shape[1]))

    iteracoes = 0

    for n_iter in range(max_iter):

        iteracoes += 1 

        dif = np.abs(Y - Y_prev).sum()
        print(dif)

        if dif < epsilon:
            break

        Y_prev = Y

        Y = safe_sparse_dot(T,Y)

        Y = compara_original(Y, dataref) # Restoring to the original value, in case it's not missing
    
    print("Number of iterations: ", iteracoes)

    return(Y)


# Interquartile range filter function
def remover_outliers_iqr(df):
  """
  Removes outliers in each column of a Pandas DataFrame using IQR.

  Arguments:
    df: Pandas DataFrame.

  Returns:
    Pandas DataFrame with outliers replaced by np.nan.
  """

  for coluna in df.columns:
    # Calculate the first and third quartiles
    q1 = df[coluna].quantile(0.25)
    q3 = df[coluna].quantile(0.75)

    # Calculate IQR
    iqr = q3 - q1

    # Define lower and upper limits
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    # Replace outliers with np.nan
    df.loc[df[coluna] < limite_inferior, coluna] = np.nan
    df.loc[df[coluna] > limite_superior, coluna] = np.nan

  return df



# Function to check columns with 100% null values. Fills with garbage instead of deleting the column during filling.

def preencher_valores_nulos(dataframe):
  """
  Checks all columns of a DataFrame and fills null values with 0.

  Arguments:
    dataframe: The DataFrame to be analyzed.

  Returns:
    The original DataFrame with null values filled with 0.
  """

  # Checks if there are columns with 100% null values
  for coluna in dataframe.columns:
    if dataframe[coluna].isnull().all():
      # Fills null values with 0
      dataframe[coluna].fillna(0, inplace=True)

  return dataframe

# Usage example
dataframe_original = pd.DataFrame({
  'Column 1': [1, 2, np.nan],
  'Column 2': [4, 5, 6],
  'Column 3': [np.nan, np.nan, np.nan]
})


####################################################################################################
# LPMD: Label Propagation for Missing Data Imputation
####################################################################################################
#
# Initial imputation with 0
#
# WITH class division in Label Propagation
#    
# Secondary imputation by total mean (transform)
#
#########################################################

class LPMD():
    ''' Receives a dataframe with missing data
        Returns complete data
    '''
    # -------------------------------------------------------------------------------
    def __init__(self, datacomplete, iteracoes=500, epsilon=1e-32):
    # Parameters
        self.max_iter = iteracoes
        print("max iterations: ", self.max_iter)
        self.epsilon = epsilon
        print("epsilon: ", epsilon)

        # Input of complete data to calculate error
        self.data = datacomplete

    # -------------------------------------------------------------------------------
    def fit(self, datamissing, train_target):       

        # Input of data with missing values
        self.dataMiss = datamissing

        # Input of target (train only)
        self.train_target = pd.DataFrame(train_target)

        # Number of clusters
        list_groups = self.train_target[0].unique()
        self.numero_clusters = len(list_groups)

        # Creating reference dataset
        self.dataref = self.dataMiss

        # Grouping using the target
        agrupamento = self.train_target[0]

        # Adding a column with cluster information in the original dataset
        self.dataMiss = pd.DataFrame(self.dataMiss)
        self.dataMiss["grupo"] = agrupamento
        Y_ok = self.dataMiss.copy(deep=True)

        # Creating a dataset to receive the imputed values
        self.dataImp = self.dataMiss.copy(deep=True)

        # Performs imputation of -1 in place of missing values
        self.dataImp.fillna(value=-1, inplace=True)
        
        # Auxiliary list to receive groups
        groups = self.dataImp["grupo"].unique()

        print("Imputed:")
        print(self.dataImp.isnull().sum())


        # Performing Label Propagation Regression for each class
        for i in groups:
           
            # Only the portion of a specific class
            df_imputed_cluster = self.dataImp.loc[self.dataImp['grupo'] == i]
            df_missing_cluster = self.dataMiss.loc[self.dataMiss['grupo'] == i]

            # Deletes cluster information
            df_imputed = df_imputed_cluster.drop(columns=["grupo"])
            df_missing = df_missing_cluster.drop(columns=["grupo"])

            # Converts to np.array
            df_imputed = df_imputed.to_numpy()
            df_missing = df_missing.to_numpy()

            # Calls Label Propagation
            retorno_lp = label_propagation(df_missing, df_imputed, self.epsilon, self.max_iter)
            
            # Converts return (np.array) to pandas dataframe
            retorno_lp = pd.DataFrame(retorno_lp)

            # Dataframe columns
            colunas = self.dataMiss.columns
            colunas = colunas[:-1] # Columns without the cluster

            # Dataset to receive input dataset only with a specific cluster
            df_entrada = Y_ok.loc[self.dataMiss['grupo'] == i]

            # Transfers index from one to another
            retorno_lp.index = df_entrada.index
            
            # Input dataset receives output from label propagation
            df_entrada[colunas] = retorno_lp

            # Output dataset receives updated information
            Y_ok.loc[self.dataMiss['grupo'] == i] = df_entrada
        
        # Removes target information
        Y_ok = Y_ok.drop(columns="grupo")

        # Filled Dataframe
        self.treino_preenchido = Y_ok

        return(self)

    # -------------------------------------------------------------------------------    
    def transform(self, datamissing, is_Train=False):

        if (is_Train):
            Y_ok = self.treino_preenchido

            # From dataframe to numpy array
            Y_ok = Y_ok.to_numpy()

        else: 
            # Input of data with missing values
            self.dataMiss = datamissing

            # Verifies the size of train data
            tamanho_dados_treino = len(self.treino_preenchido)

            # Combines train data (imputed) with test data (not imputed)
            X_sem_imputar = np.concatenate([self.treino_preenchido, self.dataMiss])
            
            # Creating reference dataset
            self.dataref =  X_sem_imputar

            # Imputation using mean
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            mean = imputer.fit(X_sem_imputar)
            self.dataImp = mean.transform(X_sem_imputar)
            
            # Grouping using fast greedy
            kmeans = KMeans(n_clusters = self.numero_clusters)
            kmeans.fit(self.dataImp)
            agrupamento = kmeans.labels_

            # Adding a column with cluster information in the original dataset
            self.dataMiss = pd.DataFrame(X_sem_imputar)
            self.dataMiss["grupo"] = agrupamento
            Y_ok = self.dataMiss.copy(deep=True)

            # Adding a column with cluster information in dataImp dataset
            self.dataImp = pd.DataFrame(self.dataImp)
            self.dataImp["grupo"] = agrupamento

            # Auxiliary list to receive groups
            groups = self.dataImp["grupo"].unique()

            # Performing Label Propagation Regression for each cluster
            for i in groups:
            
                # Only the portion of a specific class
                df_imputed_cluster = self.dataImp.loc[self.dataImp['grupo'] == i]
                df_missing_cluster = self.dataMiss.loc[self.dataMiss['grupo'] == i]

                # Deletes cluster information
                df_imputed = df_imputed_cluster.drop(columns=["grupo"])
                df_missing = df_missing_cluster.drop(columns=["grupo"])

                # Converts to np.array
                df_imputed = df_imputed.to_numpy()
                df_missing = df_missing.to_numpy()

                # Calls Label Propagation
                retorno_lp = label_propagation(df_missing, df_imputed, self.epsilon, self.max_iter)
                
                # Converts return (np.array) to pandas dataframe
                retorno_lp = pd.DataFrame(retorno_lp)

                # Dataframe columns
                colunas = self.dataMiss.columns
                colunas = colunas[:-1] # Columns without the cluster

                # Dataset to receive input dataset only with a specific cluster
                df_entrada = Y_ok.loc[self.dataMiss['grupo'] == i]

                # Transfers index from one to another
                retorno_lp.index = df_entrada.index
                
                # Input dataset receives output from label propagation
                df_entrada[colunas] = retorno_lp

                # Output dataset receives updated information
                Y_ok.loc[self.dataMiss['grupo'] == i] = df_entrada
            
            # Removes target information
            Y_ok = Y_ok.drop(columns="grupo")

            # From dataframe to numpy array
            Y_ok = Y_ok.to_numpy()

            # Slices the concatenated imputed dataset to get only test data (output_md_test)
            Y_ok = Y_ok[tamanho_dados_treino:]


        return(Y_ok)
    

#########################################################
# LPMD2: Label Propagation for Missing Data Imputation 2
#########################################################
#
# Initial imputation by class mean (FIT)
#    
# Performs filtering using interquartile range (replaces outliers with missing)
#
# WITH class division
#    
# Secondary imputation by mean (Transform)
#
# Applies the filter during PREDICT
#
#########################################################

class LPMD2():
    ''' Receives a dataframe with missing data
        Returns complete data
    '''
    # -------------------------------------------------------------------------------
    def __init__(self, datacomplete, iteracoes=500, epsilon=1e-32):
    # Parameters
        self.max_iter = iteracoes
        print("max iterations: ", self.max_iter )
        self.epsilon = epsilon
        print("epsilon: ", epsilon)

        # Input complete data for error measurement
        self.data = datacomplete

    # -------------------------------------------------------------------------------
    def fit(self, datamissing, train_target):       

        # Input data with missing values
        self.dataMiss = datamissing

        # Input target (training only)
        self.train_target = pd.DataFrame(train_target)

        # Number of clusters
        list_groups = self.train_target[0].unique()
        self.numero_clusters = len(list_groups)

        # Creating reference dataset
        self.dataref = self.dataMiss

        # Grouping using the target
        agrupamento = self.train_target[0]

        # Adding a column with cluster information to the original dataset
        self.dataMiss = pd.DataFrame(self.dataMiss)
        self.dataMiss["grupo"] = agrupamento
        Y_ok = self.dataMiss.copy(deep=True)

        # Creating a dataset to receive the imputed values
        self.dataImp = self.dataMiss.copy(deep=True)
        print("dataImp: ", self.dataImp.shape)

        # Creating an imputer object using the mean
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        
        # Auxiliary list to hold groups
        groups = self.dataImp["grupo"].unique()

        # Imputation process for each class
        for i in groups:

            # Only the portion of the specified class
            df_imputed_classe = self.dataImp.loc[self.dataImp['grupo'] == i]
            cols = df_imputed_classe.columns
            print("df_imputed_classe: ", type(df_imputed_classe))
            print(df_imputed_classe.shape)
            print(df_imputed_classe)

            # Pass through the interquartile range filter
            df_imputed_classe = remover_outliers_iqr(df_imputed_classe.copy())

            # Check if any column has 100% missing values. If so, insert placeholder values to prevent the column from disappearing
            df_imputed_classe = preencher_valores_nulos(df_imputed_classe.copy())

            # Fit the imputer
            mean = imputer.fit(df_imputed_classe)

            # Transform
            dataImp_classe = mean.transform(df_imputed_classe)
            print("Data types: ", type(dataImp_classe))
            print(dataImp_classe)

            # Assign the imputed values
            self.dataImp.loc[self.dataImp['grupo'] == i] = dataImp_classe


        # Performing Label Propagation Regression for each class
        for i in groups:
           
            # Only the portion of the specified class
            df_imputed_cluster = self.dataImp.loc[self.dataImp['grupo'] == i]
            df_missing_cluster = self.dataMiss.loc[self.dataMiss['grupo'] == i]

            # Delete cluster information
            df_imputed = df_imputed_cluster.drop(columns=["grupo"])
            df_missing = df_missing_cluster.drop(columns=["grupo"])

            # Convert to numpy array
            df_imputed = df_imputed.to_numpy()
            df_missing = df_missing.to_numpy()

            # Call Label Propagation
            retorno_lp = label_propagation(df_missing, df_imputed, self.epsilon, self.max_iter)
            
            # Convert return (np.array) to pandas dataframe
            retorno_lp = pd.DataFrame(retorno_lp)

            # Dataframe columns
            colunas = self.dataMiss.columns
            colunas = colunas[:-1] # Columns without the cluster

            # Dataset to hold input data for the specified cluster
            df_entrada = Y_ok.loc[self.dataMiss['grupo'] == i]

            # Pass the index from one to another
            retorno_lp.index = df_entrada.index
            
            # Input dataset receives output from label propagation
            df_entrada[colunas] = retorno_lp

            # Output dataset receives updated information
            Y_ok.loc[self.dataMiss['grupo'] == i] = df_entrada
        
        # Remove target information
        Y_ok = Y_ok.drop(columns="grupo")

        # Filled dataframe
        self.treino_preenchido = Y_ok

        return(self)

    # -------------------------------------------------------------------------------    
    def transform(self, datamissing, is_Train=False):

        if (is_Train):
            Y_ok = self.treino_preenchido

            # Convert from dataframe to numpy array
            Y_ok = Y_ok.to_numpy()

        else: 
            # Input data with missing values
            self.dataMiss = datamissing

            # Check the size of the training data
            tamanho_dados_treino = len(self.treino_preenchido)

            # Combine the training data (imputed) with the test data (not imputed)
            X_sem_imputar = np.concatenate([self.treino_preenchido, self.dataMiss])
            
            # Creating reference dataset
            self.dataref =  X_sem_imputar

            # Impute using MEAN
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            mean = imputer.fit(X_sem_imputar)
            self.dataImp = mean.transform(X_sem_imputar)
            
            # Perform clustering using fast greedy
            kmeans = KMeans(n_clusters = self.numero_clusters)
            kmeans.fit(self.dataImp)
            agrupamento = kmeans.labels_

            # Add a column with cluster information to the original dataset
            self.dataMiss = pd.DataFrame(X_sem_imputar)
            self.dataMiss["grupo"] = agrupamento
            Y_ok = self.dataMiss.copy(deep=True)

            # Add a column with cluster information to the dataImp dataset
            self.dataImp = pd.DataFrame(self.dataImp)
            self.dataImp["grupo"] = agrupamento

            # Auxiliary list to hold groups
            groups = self.dataImp["grupo"].unique()

            # Performing Label Propagation Regression for each cluster
            for i in groups:
            
                # Only the portion of the specified class
                df_imputed_cluster = self.dataImp.loc[self.dataImp['grupo'] == i]
                df_missing_cluster = self.dataMiss.loc[self.dataMiss['grupo'] == i]

                # Delete cluster information
                df_imputed = df_imputed_cluster.drop(columns=["grupo"])
                df_missing = df_missing_cluster.drop(columns=["grupo"])

                # Pass through the interquartile range filter
                df_imputed = remover_outliers_iqr(df_imputed.copy())

                # Fill the filtered values with the mean
                mean = imputer.fit(df_imputed)

                # Transform (imputer)
                df_imputed = mean.transform(df_imputed.copy())

                # Convert to numpy array
                # df_imputed = df_imputed.to_numpy() # When imputed, it already returns numpy
                df_missing = df_missing.to_numpy()

                # Call Label Propagation
                retorno_lp = label_propagation(df_missing, df_imputed, self.epsilon, self.max_iter)
                
                # Convert return (np.array) to pandas dataframe
                retorno_lp = pd.DataFrame(retorno_lp)

                # Dataframe columns
                colunas = self.dataMiss.columns
                colunas = colunas[:-1] # Columns without the cluster

                # Dataset to hold input data for the specified cluster
                df_entrada = Y_ok.loc[self.dataMiss['grupo'] == i]

                # Pass the index from one to another
                retorno_lp.index = df_entrada.index
                
                # Input dataset receives output from label propagation
                df_entrada[colunas] = retorno_lp

                # Output dataset receives updated information
                Y_ok.loc[self.dataMiss['grupo'] == i] = df_entrada
            
            # Remove target information
            Y_ok = Y_ok.drop(columns="grupo")

            # Convert from dataframe to numpy array
            Y_ok = Y_ok.to_numpy()

            # Slice the concatenated imputed dataset to get only the test data (output_md_test)
            Y_ok = Y_ok[tamanho_dados_treino:]


        return(Y_ok)

