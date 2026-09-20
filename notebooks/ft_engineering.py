import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def preprocesar_datos(df, num_cols, cat_cols, ord_cols):
    """Crea el pipeline y retorna los features transformados como DataFrame"""

    # Transformador numérico
    num_transformer = SimpleImputer(strategy='median')

    # Transformador categórico nominal
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # sparse_output=False es obligatorio para poder convertirlo a DataFrame después
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Transformador categórico ordinal
    ord_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])

    # Ensamblar en ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols),
            ('ord', ord_transformer, ord_cols)
        ])

    # 1. Aplicar la transformación (esto devuelve una matriz de NumPy)
    X_transformado = preprocessor.fit_transform(df)

    # 2. Extraer los nombres de todas las columnas (incluyendo las nuevas generadas por OneHot)
    nombres_columnas = preprocessor.get_feature_names_out()

    # 3. Reconstruir el DataFrame de Pandas
    df_transformado = pd.DataFrame(X_transformado, columns=nombres_columnas, index=df.index)

    return df_transformado, preprocessor
