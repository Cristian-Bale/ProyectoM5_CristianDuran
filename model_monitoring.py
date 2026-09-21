import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon

class DriftDetector:
    def __init__(self, df_referencia, df_actual):
        """Inicializa el detector con datos históricos (entrenamiento) y nuevos (producción)."""
        self.ref = df_referencia
        self.cur = df_actual
        self.resultados = []

    def ks_test(self, variable, umbral=0.05):
        """Kolmogorov-Smirnov para variables numéricas continuas."""
        stat, p_value = ks_2samp(self.ref[variable].dropna(), self.cur[variable].dropna())
        drift = p_value < umbral
        return {"Variable": variable, "Metrica": "KS Test", "Valor": round(p_value, 4), "Drift": drift}

    def chi_square_test(self, variable, umbral=0.05):
        """Chi-cuadrado para variables categóricas."""
        ref_counts = self.ref[variable].value_counts()
        cur_counts = self.cur[variable].value_counts()
        df_counts = pd.DataFrame({'ref': ref_counts, 'cur': cur_counts}).fillna(0)
        
        stat, p_value, dof, expected = chi2_contingency(df_counts.T)
        drift = p_value < umbral
        return {"Variable": variable, "Metrica": "Chi-Square", "Valor": round(p_value, 4), "Drift": drift}

    def psi_score(self, variable, bins=10):
        """Population Stability Index (PSI). Valores > 0.2 indican drift crítico."""
        ref_hist, bin_edges = np.histogram(self.ref[variable].dropna(), bins=bins, density=True)
        cur_hist, _ = np.histogram(self.cur[variable].dropna(), bins=bin_edges, density=True)
        
        # Se reemplazan los ceros por un valor minúsculo para evitar errores matemáticos al dividir o aplicar logaritmo
        ref_hist = np.where(ref_hist == 0, 0.0001, ref_hist)
        cur_hist = np.where(cur_hist == 0, 0.0001, cur_hist)
        
        psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
        drift = psi > 0.2
        return {"Variable": variable, "Metrica": "PSI", "Valor": round(psi, 4), "Drift": drift}

    def jensen_shannon(self, variable, bins=10):
        """Divergencia de Jensen-Shannon para comparar distribuciones de probabilidad."""
        ref_hist, bin_edges = np.histogram(self.ref[variable].dropna(), bins=bins, density=True)
        cur_hist, _ = np.histogram(self.cur[variable].dropna(), bins=bin_edges, density=True)
        
        js_div = jensenshannon(ref_hist, cur_hist)
        drift = js_div > 0.1
        return {"Variable": variable, "Metrica": "Jensen-Shannon", "Valor": round(js_div, 4), "Drift": drift}

    def evaluar_dataset(self, num_vars, cat_vars):
        """Ejecuta todas las métricas de monitoreo y devuelve un reporte tabular."""
        for col in num_vars:
            self.resultados.append(self.ks_test(col))
            self.resultados.append(self.psi_score(col))
            self.resultados.append(self.jensen_shannon(col))
        for col in cat_vars:
            self.resultados.append(self.chi_square_test(col))
            
        return pd.DataFrame(self.resultados)