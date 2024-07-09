import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class Data:
    def __init__(self, data = None, descrip_entrenamiento = None, descrip_prueba = None, obj_entrenamiento = None, obj_prueba = None):
        '''
        Inicializa un objeto de la clase con los parámetros dados.
    
        Parameters
        ----------
        data : csv 
            Los datos que serán utilizados en el objeto. El valor por defecto es None.
        descrip_entrenamiento : 
            La descripción del conjunto de entrenamiento. El valor por defecto es None.
        descrip_prueba : 
            La descripción del conjunto de prueba. El valor por defecto es None.
        obj_entrenamiento : 
            El objetivo del conjunto de entrenamiento. El valor por defecto es None.
        obj_prueba : 
            El objetivo del conjunto de prueba. El valor por defecto es None.
    
        Returns
        -------
        None
        '''
        self.data = data
        self.descrip_entrenamiento = descrip_entrenamiento
        self.descrip_prueba = descrip_prueba
        self.obj_entrenamiento = obj_entrenamiento
        self.obj_prueba = obj_prueba
    
    def cargar_data_Egipto(self):
        '''
        Carga la base de datos de Egipto desde un archivo CSV en la ruta especificada.
    
        Parameters
        ----------
        ruta : str
            La ruta del archivo CSV desde donde se cargarán los datos.
    
        Returns
        -------
        None
        '''
        self.data = pd.read_csv('data/Egypt_Houses_Price.csv')
        
    def cargar_data_Paper(self):
        '''
        Carga la base de datos del paper desde un archivo CSV en la ruta especificada.
    
        Parameters
        ----------
        ruta : str
            La ruta del archivo CSV desde donde se cargarán los datos.
    
        Returns
        -------
        None
        '''
        self.data = pd.read_csv('../data/Egypt_Houses_Price.csv')
        
    def get_data(self):
        '''
        Devuelve las primeras filas de los datos cargados.
    
        Parameters
        ----------
        None
    
        Returns
        -------
        pandas.DataFrame
            Un DataFrame que contiene las primeras filas de los datos cargados.
        '''
        return self.data.head(15)
    
    def __str__(self):
        '''
        Devuelve una cadena de caracteres que representa las dimensiones de los datos y la cantidad de valores nulos por columna.
    
        Parameters
        ----------
        None
    
        Returns
        -------
        str
            Una cadena de caracteres que contiene las dimensiones de los datos y la cantidad de valores nulos por columna.
        '''
        dimensiones = self.data.shape
        Nans = self.data.isnull().sum()
        return f'Dimensiones: {dimensiones} \nCantidad de nulos por columna: \n{Nans}'
    
    def eliminar_columnas(self, columnas_eliminar):
        '''
        Elimina columnas específicas del conjunto de datos.
    
        Parameters
        ----------
        columnas_eliminar : list
            Lista de nombres de columnas a eliminar del conjunto de datos.
    
        Returns
        -------
        None
        '''
        self.data = self.data.drop(columns = columnas_eliminar)
        
    def convertir_columnas_descriptivas_a_numericas(self, columnas_descriptivas):
        '''
        Convierte columnas descriptivas a tipo numérico en el conjunto de datos.
    
        Parameters
        ----------
        columnas_descriptivas : list
            Lista de nombres de columnas que se convertirán a tipo numérico.
    
        Returns
        -------
        None
        '''
        for col in columnas_descriptivas:
            self.data[col] = pd.to_numeric(self.data[col], errors = 'coerce')
            
    def eliminar_unknown(self, columna):
        '''
        Elimina las filas del conjunto de datos donde la columna especificada tiene el valor 'Unknown'.
    
        Parameters
        ----------
        columna : str
            Nombre de la columna que se verificará para eliminar filas con valor 'Unknown'.
    
        Returns
        -------
        None
        '''
        self.data = self.data[self.data[columna] != 'Unknown']
        
    def imputar_por_grupo(self, grupo_cols, col_imputar, metodo = 'mean'):
        '''
        Imputa valores faltantes en una columna específica utilizando estadísticas del grupo definido por columnas.
    
        Parameters
        ----------
        grupo_cols : list
            Lista de nombres de columnas para agrupar los datos.
        col_imputar : str
            Nombre de la columna en la que se imputarán los valores faltantes.
        metodo : str
            Método estadístico a utilizar para la imputación ('mean', 'median', 'mode'). El valor por defecto es 'mean'.
    
        Returns
        -------
        None
        '''
        grupo_stats = self.data.groupby(grupo_cols)[col_imputar].transform(metodo)
        self.data[col_imputar] = np.where(self.data[col_imputar].isna(), grupo_stats, self.data[col_imputar])
        
    def limpiar_data(self):
        '''
        Limpia la base de datos a utilizar.
        
        Parameters 
        ----------
        None
        
        Returns
        -------
        None
        '''
        # Eliminamos la columnas innecesarias
        self.eliminar_columnas(columnas_eliminar = ['Furnished', 'Level', 'Compound', 'Payment_Option', 'Delivery_Date', 'Delivery_Term'])
        # Nos aseguramos que las columnas descriptivas sean númericas
        self.convertir_columnas_descriptivas_a_numericas(columnas_descriptivas = ['Price', 'Bedrooms', 'Bathrooms', 'Area'])
        # Eliminamos las observaciones que sean desconocidas en 'Type'
        self.eliminar_unknown(columna = 'Type')
        # Imputa los valores nulos
        self.imputar_por_grupo(grupo_cols = ['Type', 'City'], col_imputar = 'Bedrooms')
        self.imputar_por_grupo(grupo_cols = ['Type', 'City'], col_imputar = 'Bathrooms')
        self.imputar_por_grupo(grupo_cols = ['Type', 'City'], col_imputar = 'Area')
        self.imputar_por_grupo(grupo_cols = ['Type', 'City'], col_imputar = 'Price')
        self.imputar_por_grupo(grupo_cols = ['Type'], col_imputar = 'Bedrooms')
        self.imputar_por_grupo(grupo_cols = ['Type'], col_imputar = 'Bathrooms')
        self.imputar_por_grupo(grupo_cols = ['Type'], col_imputar = 'Area')
        self.eliminar_columnas(columnas_eliminar = 'City')
        
    def preparar_data(self, col_objetivo, test_size = 0.25, random_state = None):
        '''
        Prepara los datos para el modelado, separando variables descriptivas y objetivo, dividiendo en conjuntos de entrenamiento y prueba,
        convirtiendo variables categóricas en variables ficticias, y escalando variables numéricas.
     
        Parameters
        ----------
        col_objetivo : str
            Nombre de la columna que representa la variable objetivo.
        test_size : float
            Proporción del conjunto de datos que se reservará para el conjunto de prueba. El valor por defecto es 0.25.
        random_state : int or None
            Semilla para la generación de números aleatorios para la división de datos. El valor por defecto es None.
     
        Returns
        -------
        None
        '''
        # Aislamos nuestra columna objetivo de nuestras columnas explicativas
        variables_descriptivas = self.data.drop(columns=[col_objetivo])
        variable_objetivo = self.data[col_objetivo]
        
        # Separar las columnas numéricas y categóricas
        variables_num = variables_descriptivas.select_dtypes(include = ['number']).columns.tolist()
        variables_cat = variables_descriptivas.select_dtypes(exclude = ['number']).columns.tolist()
        
        # Dividimos los datos de entrenamiento y prueba
        self.descrip_entrenamiento, self.descrip_prueba, self.obj_entrenamiento, self.obj_prueba = train_test_split(variables_descriptivas, variable_objetivo, test_size = test_size, random_state = random_state)
        
        # Convertir variables categóricas en dummies
        self.descrip_entrenamiento = pd.get_dummies(self.descrip_entrenamiento, columns = variables_cat, dtype = float)
        self.descrip_prueba = pd.get_dummies(self.descrip_prueba, columns = variables_cat, dtype = float)
        
        # Escalar Variable númericas
        scaler = StandardScaler()
        self.descrip_entrenamiento[variables_num] = scaler.fit_transform(self.descrip_entrenamiento[variables_num])
        self.descrip_prueba[variables_num] = scaler.transform(self.descrip_prueba[variables_num])    
        
class Graficos(Data):
    def __init__(self, data=None, descrip_entrenamiento=None, descrip_prueba=None, obj_entrenamiento=None, obj_prueba=None):
        '''
        Inicializa un objeto de la clase con los parámetros dados.
    
        Parameters
        ----------
        data : csv 
            Los datos que serán utilizados en el objeto. El valor por defecto es None.
        descrip_entrenamiento : 
            La descripción del conjunto de entrenamiento. El valor por defecto es None.
        descrip_prueba : 
            La descripción del conjunto de prueba. El valor por defecto es None.
        obj_entrenamiento : 
            El objetivo del conjunto de entrenamiento. El valor por defecto es None.
        obj_prueba : 
            El objetivo del conjunto de prueba. El valor por defecto es None.
    
        Returns
        -------
        None
        '''
        super().__init__(data, descrip_entrenamiento, descrip_prueba, obj_entrenamiento, obj_prueba)

    def matriz_correlacion(self):
        '''
        Grafica una matriz de correlación entre una variable objetivo y las demás variables cuantitativas.
        
        Parameters
        ----------
        None
        
        Returns 
        -------
        None
        '''
        df = self.data.drop(columns='Type')
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Matriz de Correlación")
        plt.show()

    def grafico_dispersion(self, x, y):
        '''
        Muuestra un gráfico de dispersión comparando una variable cuantitativa (y) con otra variable cuantitativa. 
        (x)
        Parameters 
        ----------
        x: str
            Variable en eje x 
        y: str
            Variable en eje y
            
        Returns
        -------
        None
        '''
        plt.figure(figsize=(20, 6))
        sns.scatterplot(x=self.data[x], y=self.data[y])
        plt.title(f"Gráfico de Dispersión: {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def grafico_boxplot(self, columna):
        '''
        Muestra un gráfico boxplot para una columna de la base de datos.
        
        Parameters 
        ----------
        columna: str
            Columna que se quiere graficar
        
        Returns
        -------
        None
        '''
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data, y=columna)
        plt.title(f"Boxplot: {columna}")
        plt.ylabel(columna)
        plt.show()

    def histograma(self, columna, bins=30):
        '''
        Muestra un histograma de una columna específica.
        
        Parameters 
        ----------
        columna: str
            Columna que se quiere graficar
        bins: int
            Cantidad de "barras" que se quieren ver en el histograma
        
        Returns
        -------
        None
        '''
        plt.figure(figsize=(20, 6))
        sns.histplot(self.data[columna], bins=bins, kde=True)
        plt.title(f"Histograma: {columna}")
        plt.xlabel(columna)
        plt.ylabel("Frecuencia")
        plt.show()

    
class Modelo(Graficos):
    def __init__(self, data = None, descrip_entrenamiento = None, descrip_prueba = None, obj_entrenamiento = None, obj_prueba = None):
        '''
        Inicializa un objeto de la clase con los parámetros dados y llama al método __init__ de la clase Data.
    
        Parameters
        ----------
        data : csv
            Los datos que serán utilizados en el objeto. El valor por defecto es None.
        descrip_entrenamiento : 
            La descripción del conjunto de entrenamiento. El valor por defecto es None.
        descrip_prueba : 
            La descripción del conjunto de prueba. El valor por defecto es None.
        obj_entrenamiento : 
            El objetivo del conjunto de entrenamiento. El valor por defecto es None.
        obj_prueba : 
            El objetivo del conjunto de prueba. El valor por defecto es None.
    
        Returns
        -------
        None
        '''
        super().__init__(data, descrip_entrenamiento, descrip_prueba, obj_entrenamiento, obj_prueba)

    def Modelo_Ridge(self, grado=2, alphas=[0.01, 0.1, 1, 10, 100]):
        '''
        Entrena un modelo de regresión Ridge utilizando características polinomiales y búsqueda de hiperparámetros por cuadrícula.
    
        Parameters
        ----------
        grado : int
            El grado de las características polinomiales. El valor por defecto es 2.
        alphas : list
            Lista de valores de alphas para la búsqueda en cuadrícula del mejor hiperparámetro. El valor por defecto es [0.01, 0.1, 1, 10, 100].
    
        Returns
        -------
        None
    
        '''
        
        caracteristicas_polinomiales = PolynomialFeatures(degree=grado)
        descrip_entrenamiento_polinomial = caracteristicas_polinomiales.fit_transform(self.descrip_entrenamiento)
        descrip_prueba_polinomial = caracteristicas_polinomiales.transform(self.descrip_prueba)
        param_grid = {'alpha': alphas}
        
        # Se crea el modelo
        ridge_model = Ridge()
        
        # Se realiza una búsqueda en cuadrícula para encontrar los mejores hiperparámetros
        grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(descrip_entrenamiento_polinomial, self.obj_entrenamiento)
        
        # Se obtienen los mejores hiperparámetros encontrados
        best_alpha = grid_search.best_params_['alpha']
        
        # Entrenamos el modelo con los mejores hiperparámetros
        ridge_model = Ridge(alpha=best_alpha)
        ridge_model.fit(descrip_entrenamiento_polinomial, self.obj_entrenamiento)
        
        # Realizamos las predicciones en el conjunto de prueba
        obj_prueba_predict = ridge_model.predict(descrip_prueba_polinomial)
        
        # Se calcula RMSE para el conjunto de prueba
        rmse_test = np.sqrt(mean_squared_error(self.obj_prueba, obj_prueba_predict))

        # Se calcula R-squared para el conjunto de prueba
        r2_test = r2_score(self.obj_prueba, obj_prueba_predict)

        # Se calcula el R-squared ajustado para el conjunto de prueba
        n = len(self.obj_prueba)
        p = descrip_entrenamiento_polinomial.shape[1]
        adjusted_r2_test = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)

        # Se calcula el Error Absoluto Medio (MAE) para el conjunto de prueba
        mae_test = mean_absolute_error(self.obj_prueba, obj_prueba_predict)

        # Se imprimen métricas de evaluación
        print(f'RMSE (Prueba): {rmse_test}')
        print(f'R-squared (Prueba): {r2_test}')
        print(f'R-squared ajustado (Prueba): {adjusted_r2_test}')
        print(f'Error Absoluto Medio (MAE) (Prueba): {mae_test}')

    def Modelo_regresion_polinomial(self, grado = 2):
        '''
        Entrena un modelo de regresión lineal utilizando características polinomiales.
    
        Parameters
        ----------
        grado : int
            El grado de las características polinomiales. El valor por defecto es 2.
    
        Returns
        -------
        None
        '''
        caracteristicas_polinomiales = PolynomialFeatures(degree = grado)
        descrip_entrenamiento_polinomial = caracteristicas_polinomiales.fit_transform(self.descrip_entrenamiento)
        descrip_prueba_polinomial = caracteristicas_polinomiales.fit_transform(self.descrip_prueba)
        
        # Creamos el modelo
        modelo = LinearRegression()
        # Entrenamos el modelo
        modelo.fit(descrip_entrenamiento_polinomial, self.obj_entrenamiento)
        # Realizamos las predicciones del conjunto prueba
        obj_prueba_predict = modelo.predict(descrip_prueba_polinomial)
        
        # Se calcula RMSE para el conjunto de prueba
        rmse_test = np.sqrt(mean_squared_error(self.obj_prueba, obj_prueba_predict))
        
        # Se calcula R-squared para el conjunto de prueba
        r2_test = r2_score(self.obj_prueba, obj_prueba_predict)

        # Se calcula el R-squared ajustado para el conjunto de prueba
        n = len(self.obj_prueba)
        p = descrip_entrenamiento_polinomial.shape[1]
        adjusted_r2_test = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
        
        # Se calcula el Error Absoluto Medio (MAE) para el conjunto de prueba
        mae_test = mean_absolute_error(self.obj_prueba, obj_prueba_predict)
        
        # Se imprime métricas de evaluación
        print(f'RMSE (Prueba): {rmse_test}')
        print(f'R-squared (Prueba): {r2_test}')
        print(f'R-squared ajustado (Prueba): {adjusted_r2_test}')
        print(f'Error Absoluto Medio (MAE) (Prueba): {mae_test}')
    
    def Modelo_regresion_elastic_net(self, alpha = 1.0, l1_ratio = 0.5):
        '''
        Entrena un modelo de regresión Elastic Net y evalúa su desempeño en un conjunto de prueba.
    
        Parameters
        ----------
        alpha : float
            La constante que multiplica el término de regularización. El valor por defecto es 1.0.
        l1_ratio : float
            La relación entre la penalización L1 y L2. El valor por defecto es 0.5.
    
        Returns
        -------
        None
        '''
        # Crear el modelo ElasticNet con los parámetros como palabras clave
        modelo = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        
        # Entrenar el modelo
        modelo.fit(self.descrip_entrenamiento, self.obj_entrenamiento)
        
        # Realizar las predicciones del conjunto de prueba
        obj_prueba_predict = modelo.predict(self.descrip_prueba)
        
        # Calcular RMSE para el conjunto de prueba
        rmse_test = np.sqrt(mean_squared_error(self.obj_prueba, obj_prueba_predict))
        
        # Calcular R-squared para el conjunto de prueba
        r2_test = r2_score(self.obj_prueba, obj_prueba_predict)
        
        # Calcular el R-squared ajustado para el conjunto de prueba
        n = len(self.obj_prueba)
        p = self.descrip_prueba.shape[1]
        adjusted_r2_test = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
        
        # Calcular el Error Absoluto Medio (MAE) para el conjunto de prueba
        mae_test = mean_absolute_error(self.obj_prueba, obj_prueba_predict)
        
        # Imprimir métricas de evaluación
        print(f'RMSE (Prueba): {rmse_test}')
        print(f'R-squared (Prueba): {r2_test}')
        print(f'R-squared ajustado (Prueba): {adjusted_r2_test}')
        print(f'Error Absoluto Medio (MAE) (Prueba): {mae_test}')
    
    def Modelo_regresion_lineal(self):
        '''
        Entrena un modelo de regresión lineal y evalúa su desempeño en un conjunto de prueba.
    
        Parameters
        ----------
        None
    
        Returns
        -------
        None
        '''
        # Creamos el modelo
        modelo = LinearRegression()
        # Entrenamos el modelo
        modelo.fit(self.descrip_entrenamiento, self.obj_entrenamiento)
        # Realizamos las predicciones en el conjunto de prueba
        obj_prueba_predict = modelo.predict(self.descrip_prueba)
        
        # Se calcula RMSE para el conjunto de prueba
        rmse_test = np.sqrt(mean_squared_error(self.obj_prueba, obj_prueba_predict))

        # Se calcula R-squared para el conjunto de prueba
        r2_test = r2_score(self.obj_prueba, obj_prueba_predict)

        # Se calcula el R-squared ajustado para el conjunto de prueba
        n = len(self.obj_prueba)
        p = self.descrip_prueba.shape[1]
        adjusted_r2_test = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
        
        # Se calcula el Error Absoluto Medio (MAE) para el conjunto de prueba
        mae_test = mean_absolute_error(self.obj_prueba, obj_prueba_predict)
        
        # Se imprime métricas de evaluación
        print(f'RMSE (Prueba): {rmse_test}')
        print(f'R-squared (Prueba): {r2_test}')
        print(f'R-squared ajustado (Prueba): {adjusted_r2_test}')
        print(f'Error Absoluto Medio (MAE) (Prueba): {mae_test}')
    
    def Modelo_Lasso(self, alphas = [0.01, 0.1, 1, 10, 100]):
        '''
        Entrena un modelo de regresión Lasso utilizando búsqueda de hiperparámetros por cuadrícula.
    
        Parameters
        ----------
        alphas : list
            Lista de valores de alpha para la búsqueda en cuadrícula del mejor hiperparámetro. El valor por defecto es [0.01, 0.1, 1, 10, 100].
    
        Returns
        -------
        None
        '''
        param_grid = {'alpha': alphas}
        
        # Creamos el modelo
        lasso_model = Lasso()
        
        # Realizamos la búsqueda en cuadrícula para encontrar el mejor hiperparámetro alpha
        grid_search = GridSearchCV(lasso_model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.descrip_entrenamiento, self.obj_entrenamiento)
        
        # Obtenemos el mejor valor de alpha
        best_alpha = grid_search.best_params_['alpha']
        
        # Entrenamos el modelo con el mejor valor de alpha
        lasso_model = Lasso(alpha=best_alpha)
        lasso_model.fit(self.descrip_entrenamiento, self.obj_entrenamiento)
        
        # Realizamos las predicciones del conjunto de prueba
        obj_prueba_predict = lasso_model.predict(self.descrip_prueba)
        
        # Calculamos RMSE para el conjunto de prueba
        rmse_test = np.sqrt(mean_squared_error(self.obj_prueba, obj_prueba_predict))

        # Calculamos R-squared para el conjunto de prueba
        r2_test = r2_score(self.obj_prueba, obj_prueba_predict)

        # Calculamos el R-squared ajustado para el conjunto de prueba
        n = len(self.obj_prueba)
        p = self.descrip_prueba.shape[1]
        adjusted_r2_test = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
        
        # Calculamos el Error Absoluto Medio (MAE) para el conjunto de prueba
        mae_test = mean_absolute_error(self.obj_prueba, obj_prueba_predict)
        
        # Imprimimos métricas de evaluación
        print(f'RMSE (Prueba): {rmse_test}')
        print(f'R-squared (Prueba): {r2_test}')
        print(f'R-squared ajustado (Prueba): {adjusted_r2_test}')
        print(f'Error Absoluto Medio (MAE) (Prueba): {mae_test}')  
        