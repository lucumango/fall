import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_PATH = "/Users/lonelynode/fall/MOD2 Alumnos QHub Otoño 2026 - Consolidado.csv"
OUT_DIR   = "/Users/lonelynode/fall/"
PALETTE   = "Set2"
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)

# ─────────────────────────────────────────────
#  1. CARGA Y LIMPIEZA
# ─────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH)

# la columna 'age' aparece dos veces; pandas la renombra automáticamente a age.1
# Usamos la versión numérica (age) y eliminamos duplicados / columnas basura
df = df_raw.copy()
df.columns = [c.strip() for c in df.columns]

# Si existe age.1 la descartamos (es la misma que age)
if "age.1" in df.columns:
    df.drop(columns=["age.1"], inplace=True)

# Eliminar columnas de texto libre que no aportan al análisis
drop_cols = ["email", "name", "mobil", "birth", "consent", "birth_date"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Limpiar 'age' a numérico
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce")
df["comfort"] = pd.to_numeric(df["comfort"], errors="coerce")
df["mod"] = pd.to_numeric(df["mod"], errors="coerce")

print("=" * 60)
print("DATASET POST-LIMPIEZA")
print("=" * 60)
print(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
print(f"\nColumnas: {list(df.columns)}")
print("\nTipos:\n", df.dtypes)
print("\nPrimeras filas:")
print(df.head(3).to_string())

# ─────────────────────────────────────────────
#  2. CALIDAD DE DATOS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CALIDAD DE DATOS")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
quality_df = pd.DataFrame({
    "Nulos": missing,
    "% Nulos": missing_pct,
    "Tipo": df.dtypes
})
print(quality_df[quality_df["Nulos"] > 0])
if quality_df[quality_df["Nulos"] > 0].empty:
    print("  Sin valores nulos en columnas seleccionadas.")

dup = df.duplicated().sum()
print(f"\nDuplicados completos: {dup}")
print(f"\nResumen numérico:")
print(df.describe().T.round(2).to_string())

# ─────────────────────────────────────────────
#  3. GRAFICOS EDA
# ─────────────────────────────────────────────

# -- 3.1  Distribución de variables categóricas clave ---
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Distribución de Variables Categóricas", fontsize=15, fontweight="bold")

# Gender
gender_counts = df["gender"].str.lower().map({"f": "Femenino", "m": "Masculino"}).value_counts()
axes[0, 0].pie(gender_counts, labels=gender_counts.index,
               autopct="%1.1f%%", colors=sns.color_palette(PALETTE),
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[0, 0].set_title("Distribución por Género")

# Study level
study_counts = df["study"].value_counts()
short_labels = [s.replace("Estudiante de ", "").replace(" (universidad)", "").replace("Estudiante en ", "") for s in study_counts.index]
axes[0, 1].barh(short_labels, study_counts.values, color=sns.color_palette(PALETTE, len(study_counts)))
axes[0, 1].set_xlabel("Cantidad")
axes[0, 1].set_title("Nivel de Estudio")
axes[0, 1].invert_yaxis()

# City
city_counts = df["city"].value_counts().head(10)
axes[1, 0].barh(city_counts.index, city_counts.values, color=sns.color_palette("muted", len(city_counts)))
axes[1, 0].set_xlabel("Cantidad")
axes[1, 0].set_title("Top 10 Ciudades")
axes[1, 0].invert_yaxis()

# Age group
ag_counts = df["age_group"].value_counts()
axes[1, 1].bar(ag_counts.index, ag_counts.values, color=sns.color_palette(PALETTE, len(ag_counts)))
axes[1, 1].set_xlabel("Grupo de Edad")
axes[1, 1].set_ylabel("Cantidad")
axes[1, 1].set_title("Grupos de Edad")
axes[1, 1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(OUT_DIR + "fig1_categoricas.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[OK] fig1_categoricas.png guardado")

# -- 3.2  Variables numéricas: distribuciones ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Distribución de Variables Numéricas", fontsize=15, fontweight="bold")

for ax, col, color in zip(axes, ["age", "comfort", "birth_year"],
                          sns.color_palette(PALETTE, 3)):
    data = df[col].dropna()
    ax.hist(data, bins=12, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Media {data.mean():.1f}")
    ax.axvline(data.median(), color="navy", linestyle=":", linewidth=1.5, label=f"Mediana {data.median():.1f}")
    ax.set_title(f"Distribución: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR + "fig2_numericas.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] fig2_numericas.png guardado")

# -- 3.3  Área de estudio vs. Comfort (boxplot) ---
fig, ax = plt.subplots(figsize=(14, 6))
order = df.groupby("area_grouped")["comfort"].median().sort_values(ascending=False).index
short_area = {a: a.split("/")[0].strip()[:30] for a in order}
df_plot = df.copy()
df_plot["area_short"] = df_plot["area_grouped"].map(short_area)
sns.boxplot(data=df_plot, x="area_short", y="comfort", order=[short_area[k] for k in order],
            palette=PALETTE, ax=ax)
ax.set_title("Nivel de Comodidad (comfort) por Área de Estudio", fontsize=13, fontweight="bold")
ax.set_xlabel("Área")
ax.set_ylabel("Comfort (1-10)")
ax.tick_params(axis="x", rotation=35)
plt.tight_layout()
plt.savefig(OUT_DIR + "fig3_comfort_area.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] fig3_comfort_area.png guardado")

# -- 3.4  Género vs. Comfort y Nivel de Estudio ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Género vs. Comfort y Nivel de Estudio", fontsize=14, fontweight="bold")

df_g = df.copy()
df_g["gender_label"] = df_g["gender"].str.lower().map({"f": "Femenino", "m": "Masculino"})

sns.violinplot(data=df_g, x="gender_label", y="comfort",
               palette={"Femenino": "#E8A598", "Masculino": "#89B4D1"},
               inner="box", ax=axes[0])
axes[0].set_title("Comfort por Género")
axes[0].set_xlabel("")
axes[0].set_ylabel("Comfort")

# Stacked bar: study × gender
pivot = df_g.groupby(["gender_label", "study"]).size().unstack(fill_value=0)
pivot.plot(kind="bar", stacked=True, ax=axes[1], colormap="Pastel1", edgecolor="white")
axes[1].set_title("Nivel de Estudio por Género")
axes[1].set_xlabel("")
axes[1].set_ylabel("Cantidad")
axes[1].tick_params(axis="x", rotation=15)
axes[1].legend(fontsize=6, loc="upper right",
               labels=[s.replace("Estudiante de ", "")[:25] for s in pivot.columns])
plt.tight_layout()
plt.savefig(OUT_DIR + "fig4_genero.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] fig4_genero.png guardado")

# -- 3.5  Heatmap de correlación ---
fig, ax = plt.subplots(figsize=(8, 5))
num_cols = df.select_dtypes(include="number").columns.tolist()
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Matriz de Correlación (variables numéricas)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR + "fig5_correlacion.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] fig5_correlacion.png guardado")

# -- 3.6  Módulo vs. Comfort ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.stripplot(data=df, x="mod", y="comfort", jitter=True, palette=PALETTE, size=8, ax=ax)
sns.boxplot(data=df, x="mod", y="comfort", width=0.3, color="white",
            boxprops={"alpha": 0.6}, ax=ax, fliersize=0)
ax.set_title("Comfort por Módulo", fontsize=13, fontweight="bold")
ax.set_xlabel("Módulo")
ax.set_ylabel("Comfort (1-10)")
plt.tight_layout()
plt.savefig(OUT_DIR + "fig6_modulo_comfort.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] fig6_modulo_comfort.png guardado")

# -- 3.7  Ciudad de origen mapa de palabras (barras top 15) ---
fig, ax = plt.subplots(figsize=(12, 6))
city_all = df["city"].value_counts().head(15)
bars = ax.barh(city_all.index[::-1], city_all.values[::-1],
               color=sns.color_palette("viridis", len(city_all)))
for bar, val in zip(bars, city_all.values[::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=9)
ax.set_title("Top 15 Ciudades de Procedencia", fontsize=13, fontweight="bold")
ax.set_xlabel("Alumnos")
plt.tight_layout()
plt.savefig(OUT_DIR + "fig7_ciudades.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] fig7_ciudades.png guardado")

# ─────────────────────────────────────────────
#  4. PREPARACIÓN PARA CLUSTERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PREPARACIÓN PARA CLUSTERING")
print("=" * 60)

CLUSTER_FEATURES = ["mod", "age", "comfort"]
cat_for_cluster  = ["gender", "study", "area_grouped", "age_group"]

df_cluster = df[CLUSTER_FEATURES + cat_for_cluster].copy()

# Encoders para categóricas
le_map = {}
for col in cat_for_cluster:
    le = LabelEncoder()
    df_cluster[col + "_enc"] = le.fit_transform(df_cluster[col].astype(str))
    le_map[col] = le

feature_cols = CLUSTER_FEATURES + [c + "_enc" for c in cat_for_cluster]
print(f"Features usadas: {feature_cols}")

X_raw = df_cluster[feature_cols].values

# Imputar nulos con mediana
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X_raw)

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

print(f"\nMatriz final: {X_scaled.shape[0]} filas × {X_scaled.shape[1]} features")
print(f"Nulos antes de imputar: {np.isnan(X_raw).sum()}")

# ─────────────────────────────────────────────
#  5. ELBOW + SILHOUETTE (K-Means)
# ─────────────────────────────────────────────
k_range = range(2, min(10, len(df_cluster) // 3))
inertias, sil_scores, ch_scores, db_scores = [], [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Selección del Número de Clusters (K-Means)", fontsize=14, fontweight="bold")

axes[0, 0].plot(list(k_range), inertias, "o-", color="#E55B4D", linewidth=2)
axes[0, 0].set_title("Método del Codo (Inercia)")
axes[0, 0].set_xlabel("k"); axes[0, 0].set_ylabel("Inercia")

axes[0, 1].plot(list(k_range), sil_scores, "o-", color="#4CAF93", linewidth=2)
axes[0, 1].axvline(x=list(k_range)[np.argmax(sil_scores)], color="red", linestyle="--", alpha=0.7)
axes[0, 1].set_title("Silhouette Score")
axes[0, 1].set_xlabel("k"); axes[0, 1].set_ylabel("Score")

axes[1, 0].plot(list(k_range), ch_scores, "o-", color="#5B9BD5", linewidth=2)
axes[1, 0].set_title("Calinski-Harabasz Score")
axes[1, 0].set_xlabel("k"); axes[1, 0].set_ylabel("Score")

axes[1, 1].plot(list(k_range), db_scores, "o-", color="#F5A623", linewidth=2)
axes[1, 1].axvline(x=list(k_range)[np.argmin(db_scores)], color="red", linestyle="--", alpha=0.7)
axes[1, 1].set_title("Davies-Bouldin Score (↓ mejor)")
axes[1, 1].set_xlabel("k"); axes[1, 1].set_ylabel("Score")

plt.tight_layout()
plt.savefig(OUT_DIR + "fig8_elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] fig8_elbow_silhouette.png guardado")

best_k = list(k_range)[np.argmax(sil_scores)]
print(f"\nMejor k según Silhouette: {best_k}  (score={max(sil_scores):.3f})")
print(f"k según Davies-Bouldin más bajo: {list(k_range)[np.argmin(db_scores)]}")

# ─────────────────────────────────────────────
#  6. MODELO FINAL K-MEANS  +  PCA
# ─────────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=30)
km_labels = km_final.fit_predict(X_scaled)
df["cluster_kmeans"] = km_labels

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"K-Means k={best_k} — Proyección PCA", fontsize=14, fontweight="bold")

palette_c = sns.color_palette(PALETTE, best_k)

for c in range(best_k):
    mask = km_labels == c
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=f"Cluster {c}", color=palette_c[c], s=80, edgecolors="white", linewidths=0.5)
axes[0].set_title(f"PCA (var explicada: {sum(var_exp)*100:.1f}%)")
axes[0].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
axes[0].legend()

# Centroides en espacio PCA
centers_pca = pca.transform(km_final.cluster_centers_)
axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1],
                c="black", marker="X", s=180, zorder=5, label="Centroides")
axes[0].legend()

# Tamaño de cada cluster
cluster_counts = pd.Series(km_labels).value_counts().sort_index()
axes[1].bar([f"Cluster {i}" for i in cluster_counts.index],
            cluster_counts.values,
            color=palette_c)
for i, v in enumerate(cluster_counts.values):
    axes[1].text(i, v + 0.2, str(v), ha="center", fontweight="bold")
axes[1].set_title("Tamaño de Clusters")
axes[1].set_ylabel("Alumnos")
axes[1].set_ylim(0, cluster_counts.max() * 1.2)

plt.tight_layout()
plt.savefig(OUT_DIR + "fig9_kmeans_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] fig9_kmeans_pca.png guardado")

# ─────────────────────────────────────────────
#  7. COMPARACIÓN DE ALGORITMOS
# ─────────────────────────────────────────────
results = {}

# K-Means
km_sil = silhouette_score(X_scaled, km_labels)
km_ch  = calinski_harabasz_score(X_scaled, km_labels)
km_db  = davies_bouldin_score(X_scaled, km_labels)
results["KMeans"] = {"silhouette": km_sil, "calinski_harabasz": km_ch, "davies_bouldin": km_db, "n_clusters": best_k}

# Agglomerative
for linkage in ["ward", "complete", "average"]:
    agg = AgglomerativeClustering(n_clusters=best_k, linkage=linkage)
    agg_labels = agg.fit_predict(X_scaled)
    results[f"Agglo-{linkage}"] = {
        "silhouette": silhouette_score(X_scaled, agg_labels),
        "calinski_harabasz": calinski_harabasz_score(X_scaled, agg_labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, agg_labels),
        "n_clusters": best_k
    }
    df[f"cluster_agglo_{linkage}"] = agg_labels

results_df = pd.DataFrame(results).T.round(4)
print("\nCOMPARACIÓN DE ALGORITMOS:")
print(results_df.to_string())

# Gráfico comparativo
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Comparación de Algoritmos de Clustering", fontsize=14, fontweight="bold")
colors = sns.color_palette("tab10", len(results_df))

for ax, metric, better in zip(axes,
                               ["silhouette", "calinski_harabasz", "davies_bouldin"],
                               ["↑ mayor mejor", "↑ mayor mejor", "↓ menor mejor"]):
    vals = results_df[metric].astype(float)
    bars = ax.bar(vals.index, vals.values, color=colors, edgecolor="white")
    best_idx = vals.idxmax() if "bouldin" not in metric else vals.idxmin()
    # Highlight best
    for bar, name in zip(bars, vals.index):
        if name == best_idx:
            bar.set_edgecolor("black")
            bar.set_linewidth(2.5)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + vals.max()*0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)
    ax.set_title(f"{metric}\n({better})")
    ax.tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig(OUT_DIR + "fig10_comparacion_algos.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] fig10_comparacion_algos.png guardado")

# ─────────────────────────────────────────────
#  8. PERFIL DE CLUSTERS
# ─────────────────────────────────────────────
df_profile = df.copy()
df_profile["gender_label"] = df_profile["gender"].str.lower().map({"f": "Femenino", "m": "Masculino"})

print("\n" + "=" * 60)
print("PERFIL DE CLUSTERS (K-Means)")
print("=" * 60)

for feat in ["age", "comfort", "mod"]:
    print(f"\n{feat} por cluster:")
    print(df_profile.groupby("cluster_kmeans")[feat].agg(["mean", "std", "min", "max"]).round(2).to_string())

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(f"Perfil de Clusters K-Means (k={best_k})", fontsize=14, fontweight="bold")

# Radar-style: medias de features numéricas por cluster
means = df_profile.groupby("cluster_kmeans")[["age", "comfort", "mod"]].mean()
means.T.plot(kind="bar", ax=axes[0, 0], colormap="Set2", rot=0)
axes[0, 0].set_title("Medias por Feature y Cluster")
axes[0, 0].set_ylabel("Valor promedio")
axes[0, 0].legend(title="Cluster", fontsize=8)

# Distribución de género por cluster
gender_clust = df_profile.groupby(["cluster_kmeans", "gender_label"]).size().unstack(fill_value=0)
gender_clust.plot(kind="bar", ax=axes[0, 1], colormap="Pastel1", edgecolor="white", rot=0)
axes[0, 1].set_title("Género por Cluster")
axes[0, 1].set_xlabel("Cluster")
axes[0, 1].set_ylabel("Alumnos")

# Boxplot comfort por cluster
sns.boxplot(data=df_profile, x="cluster_kmeans", y="comfort", palette=PALETTE, ax=axes[1, 0])
axes[1, 0].set_title("Comfort por Cluster")
axes[1, 0].set_xlabel("Cluster")
axes[1, 0].set_ylabel("Comfort")

# Distribución de age_group por cluster
ag_clust = df_profile.groupby(["cluster_kmeans", "age_group"]).size().unstack(fill_value=0)
ag_clust.plot(kind="bar", stacked=True, ax=axes[1, 1], colormap="tab10", edgecolor="white", rot=0)
axes[1, 1].set_title("Grupo de Edad por Cluster")
axes[1, 1].set_xlabel("Cluster")
axes[1, 1].set_ylabel("Alumnos")
axes[1, 1].legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig(OUT_DIR + "fig11_perfil_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] fig11_perfil_clusters.png guardado")

# ─────────────────────────────────────────────
#  9. RESUMEN FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN EJECUTIVO")
print("=" * 60)
print(f"""
DATASET
  Alumnos totales  : {len(df)}
  Variables usadas : {len(feature_cols)} (3 numéricas + 4 categóricas codificadas)

CALIDAD DE DATOS
  Duplicados completos : {dup}
  Nulos en features    : {np.isnan(X_raw).sum()} (imputados con mediana)

ESTADÍSTICAS CLAVE
  Edad promedio   : {df['age'].mean():.1f} años  (min {df['age'].min():.0f} - max {df['age'].max():.0f})
  Comfort promedio: {df['comfort'].mean():.2f} / 10
  Género F/M      : {(df['gender'].str.lower()=='f').sum()} / {(df['gender'].str.lower()=='m').sum()}
  Ciudad top      : {df['city'].value_counts().idxmax()} ({df['city'].value_counts().max()} alumnos)
  Área top        : {df['area_grouped'].value_counts().idxmax()[:40]}

CLUSTERING (K-Means, k={best_k})
  Silhouette Score     : {km_sil:.4f}  {'(bueno)' if km_sil > 0.4 else '(moderado)'}
  Calinski-Harabasz    : {km_ch:.2f}
  Davies-Bouldin       : {km_db:.4f}  {'(bueno)' if km_db < 1.0 else '(moderado)'}
  Varianza PCA (2 PC)  : {sum(var_exp)*100:.1f}%

DISTRIBUCIÓN POR CLUSTER:
{pd.Series(km_labels).value_counts().sort_index().to_string()}
""")

best_algo = results_df["silhouette"].astype(float).idxmax()
print(f"Mejor algoritmo por Silhouette: {best_algo} ({results_df.loc[best_algo,'silhouette']:.4f})")
print("\nFiguras generadas:")
for i, name in enumerate(["fig1_categoricas", "fig2_numericas", "fig3_comfort_area",
                           "fig4_genero", "fig5_correlacion", "fig6_modulo_comfort",
                           "fig7_ciudades", "fig8_elbow_silhouette", "fig9_kmeans_pca",
                           "fig10_comparacion_algos", "fig11_perfil_clusters"], 1):
    print(f"  {i:2}. {name}.png")
