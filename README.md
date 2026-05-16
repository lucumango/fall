# EDA — QHub Otoño 2026

Análisis exploratorio y clustering de la cohorte **MOD2 Alumnos QHub Otoño 2026** (46 alumnos, Perú).

---

## Dataset

| Campo | Detalle |
|---|---|
| Filas | 42 alumnos (solo Perú) |
| Removidos | 4 no-peruanos: Brandon Alcántara (México), Ticiana Angelucci (Argentina), José Olarte y Maria Honor (Bolivia) |
| Variables | 11 (tras limpieza) |
| Nulos | 0 |
| Duplicados | 0 |
| Fuente | `MOD2 Alumnos QHub Otoño 2026 - Consolidado.csv` |

---

## Hallazgos principales

- **Edad promedio**: 19.9 años (rango 13–27)
- **Comfort promedio**: 4.15 / 10 — nivel bajo-moderado con alta dispersión
- **Género**: 72% masculino (33) / 28% femenino (13)
- **Ciudad**: Lima Metropolitana concentra el 78% de alumnos peruanos
- **Área dominante**: Ingeniería de Software / Sistemas / Informática
- **Módulo**: mayoría en Módulo 1 (35 vs 11 en Módulo 2)
- `mod` y `comfort` prácticamente no correlacionan (r=0.08)

---

## Figuras generadas

| Figura | Descripción |
|---|---|
| `fig1_categoricas.png` | Distribución género, nivel de estudio, ciudades y grupos de edad |
| `fig2_numericas.png` | Histogramas de edad, comfort y año de nacimiento |
| `fig3_comfort_area.png` | Boxplot de comfort por área de estudio |
| `fig4_genero.png` | Violinplot comfort por género + stacked bar nivel de estudio |
| `fig5_correlacion.png` | Heatmap de correlación entre variables numéricas |
| `fig6_modulo_comfort.png` | Stripplot comfort por módulo |
| `fig7_ciudades.png` | Top 15 ciudades de procedencia |
| `fig8_elbow_silhouette.png` | Codo, Silhouette, Calinski-Harabasz y Davies-Bouldin |
| `fig9_kmeans_pca.png` | Proyección PCA de K-Means original |
| `fig10_comparacion_algos.png` | Comparación KMeans vs Agglomerative (ward/complete/average) |
| `fig11_perfil_clusters.png` | Perfil de clusters: medias, género, comfort y grupos etarios |
| `figA_diagnostico_k.png` | Diagnóstico de por qué k=9 es un artefacto métrico |
| `figB_silhouette_plot.png` | Silhouette plot por cluster (k=3 final) |
| `figC_pca_nombres.png` | PCA con nombre de cada alumno y centroides |
| `figD_tarjetas_clusters.png` | Tarjetas de perfil detallado por cluster |
| `figE_radar.png` | Radar de perfil normalizado por cluster |

---

## Clustering

### ¿Por qué no k=9?

La búsqueda automática por silhouette máximo sugirió k=9, pero con 46 alumnos eso genera clusters de 1–5 personas. Un cluster de 1 persona tiene silhouette artificialmente alta — no indica estructura real. Se eligió **k=3** por codo de inercia e interpretabilidad.

### Métricas finales (k=3, K-Means)

| Métrica | Valor |
|---|---|
| Silhouette | 0.240 |
| Calinski-Harabasz | 13.63 |
| Davies-Bouldin | 1.453 |
| Varianza explicada PCA (2 PC) | 51.2% |

Scores moderados, esperados para un dataset pequeño con variables mixtas (numéricas + categóricas codificadas) y poca separación natural entre grupos. Leve mejora respecto al análisis con los 46 alumnos originales.

### Clusters identificados

**Cluster 0 — Avanzados con Confianza Media** (n=12, silueta=0.322)
- Edad media: 20.0 años (grupo 18–24)
- Comfort: 4.3/10
- **100% Módulo 2** — el cluster más homogéneo y limpio

**Cluster 1 — Adolescentes en Transición** (n=13, silueta=0.221)
- Edad media: 16.7 años (grupo Under 18)
- Comfort: 3.8/10 — los menos cómodos del grupo
- Mayoría en Módulo 1, mezcla de secundaria y gap year

**Cluster 2 — Iniciados con Potencial** (n=17, silueta=0.196)
- Edad media: 21.9 años (grupo 18–24)
- Comfort: 4.0/10 — alta dispersión interna
- Universitarios en Módulo 1

---

## Outliers y datos curiosos

- **Alumno más joven**: Mathias Canales Díaz, **13 años** — y está en **Módulo 2**, no en el 1. El único menor de 14 en todo el grupo y ya va por el módulo avanzado.
- **Alumnos más grandes**: Dalia Zapata, Leonardo Sánchez y Luis Mejia, **27 años** los tres — en Módulo 1 con comfort bajo (1, 5 y 2 respectivamente). 14 años de diferencia con el más joven.
- **Mayor comfort del dataset**: Mixu Alarcon, 19 años, Módulo 1 — **9/10**. Outlier positiva clara; quedó en el cluster de "adolescentes" por su perfil categórico.
- **Menor comfort del dataset**: Dalia Zapata, 27 años — **1/10**. La más grande del Módulo 1 y la menos cómoda de toda la cohorte.
- **Ángel Antezana, 17 años en Módulo 2**: el más joven del módulo avanzado después de Mathias, con comfort 7/10.
- **Alejandra Caceres, 26 años, Cusco**: única persona mayor de 25 fuera de Lima, con comfort 7/10 — contrasta con el patrón general.
- **Casi todo Perú**: 42/46 alumnos son peruanos; hay 2 de Bolivia, 1 de México y 1 de Argentina — pero el 78% es de Lima Metropolitana, lo que reduce la varianza geográfica real disponible para el clustering.

---

## Scripts

| Script | Descripción |
|---|---|
| `eda_fall.py` | EDA completo: limpieza, estadísticas, 11 figuras, clustering exploratorio |
| `eda_clusters_named.py` | Diagnóstico de k óptimo, clusters con nombres, ejemplos de alumnos, 5 figuras adicionales |

```bash
python3 eda_fall.py
python3 eda_clusters_named.py
```

Requiere: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
