import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def preprocesar_datos(df, num_cols, cat_cols, ord_cols):
    """Crea el pipiline y retorna los feactures transformados"""

    #Transformador numérico
    num_transformer = SimpleImputer(strategy='median')

    #Transformador categórico nominal
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    #Trasnformador categórico ordinal
    ord_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])

    #Emsamblar en ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols),
            ('ord', ord_transformer, ord_cols)
        ])

    #Aplicar la transformación 

    X_transformado = preprocessor.fit_transform(df)

    return X_transformado, preprocessor