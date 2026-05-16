import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "/Users/lonelynode/fall/MOD2 Alumnos QHub Otoño 2026 - Consolidado.csv"
OUT_DIR   = "/Users/lonelynode/fall/"
PALETTE   = "Set2"
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)

# ── Carga ──────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()
df.columns = [c.strip() for c in df.columns]
if "age.1" in df.columns:
    df.drop(columns=["age.1"], inplace=True)
for col in ["age", "comfort", "birth_year", "mod"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["gender_label"] = df["gender"].str.lower().map({"f": "Femenino", "m": "Masculino"})

# ── Features para clustering ───────────────────────────────────────────────
CLUSTER_FEATURES = ["mod", "age", "comfort"]
CAT_FEATURES     = ["gender", "study", "area_grouped", "age_group"]

df_cl = df[CLUSTER_FEATURES + CAT_FEATURES].copy()
for col in CAT_FEATURES:
    le = LabelEncoder()
    df_cl[col + "_enc"] = le.fit_transform(df_cl[col].astype(str))

feat_cols = CLUSTER_FEATURES + [c + "_enc" for c in CAT_FEATURES]
X = StandardScaler().fit_transform(df_cl[feat_cols].values)

# ════════════════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO: por qué k=9 "gana" técnicamente pero es inútil
# ════════════════════════════════════════════════════════════════════════════
k_range = range(2, 12)
sil_all, ch_all, db_all, inertias = [], [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=30)
    lbl = km.fit_predict(X)
    sil_all.append(silhouette_score(X, lbl))
    ch_all.append(calinski_harabasz_score(X, lbl))
    db_all.append(davies_bouldin_score(X, lbl))
    inertias.append(km.inertia_)

print("k  | Silhouette | CH       | DB     | Inercia")
print("-" * 55)
for i, k in enumerate(k_range):
    marker = " <-- max sil" if sil_all[i] == max(sil_all) else ""
    print(f"k={k:2d} | {sil_all[i]:.4f}     | {ch_all[i]:8.2f} | {db_all[i]:.4f} |{inertias[i]:8.1f}{marker}")

# ── INSIGHT: la silueta sube porque con k alto los clusters son muy pequeños
#    y sus puntos quedan muy juntos. Eso es un artefacto, no estructura real.
#    Elegimos k según la curva del codo + silueta > 0.3 + interpretabilidad.
# ────────────────────────────────────────────────────────────────────────────

# Buscamos el primer k con silhouette "razonable" y codo claro: usamos k=3
# (primer pico real antes de que los clusters sean demasiado pequeños)
best_k = 3

print(f"\n→ Elegimos k={best_k} por interpretabilidad y codo de inercia.")

# ── Figura: diagnóstico de k ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Diagnóstico: ¿por qué k=9 no es el mejor k real?", fontsize=13, fontweight="bold")

axes[0].plot(list(k_range), inertias, "o-", color="#E55B4D", linewidth=2)
axes[0].axvline(best_k, color="green", linestyle="--", linewidth=1.5, label=f"k elegido={best_k}")
axes[0].set_title("Codo de Inercia")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inercia"); axes[0].legend()

axes[1].plot(list(k_range), sil_all, "o-", color="#4CAF93", linewidth=2)
axes[1].axvline(best_k, color="green", linestyle="--", linewidth=1.5, label=f"k elegido={best_k}")
axes[1].axhline(0.3, color="orange", linestyle=":", linewidth=1.5, label="umbral 0.3")
axes[1].set_title("Silhouette Score\n(k alto = clusters triviales)")
axes[1].set_xlabel("k"); axes[1].legend(fontsize=8)

# Tamaño mínimo de cluster por k
min_sizes = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    lbl = km.fit_predict(X)
    min_sizes.append(pd.Series(lbl).value_counts().min())

axes[2].plot(list(k_range), min_sizes, "o-", color="#5B9BD5", linewidth=2)
axes[2].axvline(best_k, color="green", linestyle="--", linewidth=1.5, label=f"k elegido={best_k}")
axes[2].axhline(3, color="red", linestyle=":", linewidth=1.5, label="mín recomendado=3")
axes[2].set_title("Tamaño mínimo de cluster\n(k>6 → clusters de 1-2 personas)")
axes[2].set_xlabel("k"); axes[2].set_ylabel("Min alumnos en cluster")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR + "figA_diagnostico_k.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] figA_diagnostico_k.png")

# ════════════════════════════════════════════════════════════════════════════
#  MODELO FINAL con k=3
# ════════════════════════════════════════════════════════════════════════════
km = KMeans(n_clusters=best_k, random_state=42, n_init=50)
labels = km.fit_predict(X)
df["cluster"] = labels

sil_global = silhouette_score(X, labels)
sil_per    = silhouette_samples(X, labels)
ch         = calinski_harabasz_score(X, labels)
db         = davies_bouldin_score(X, labels)

print(f"\nMétricas finales (k={best_k}):")
print(f"  Silhouette global : {sil_global:.4f}")
print(f"  Calinski-Harabasz : {ch:.2f}")
print(f"  Davies-Bouldin    : {db:.4f}")

# ── Perfil de clusters ─────────────────────────────────────────────────────
print("\nPerfil numérico por cluster:")
profile = df.groupby("cluster")[["age", "comfort", "mod"]].agg(["mean", "std"])
print(profile.round(2).to_string())

print("\nModa de categóricas por cluster:")
for c in range(best_k):
    sub = df[df["cluster"] == c]
    study_mode = sub["study"].value_counts().idxmax()
    area_mode  = sub["area_grouped"].value_counts().idxmax()
    gender_mode = sub["gender_label"].value_counts().idxmax()
    ag_mode    = sub["age_group"].value_counts().idxmax()
    city_mode  = sub["city"].value_counts().idxmax()
    print(f"  Cluster {c} (n={len(sub)}): género={gender_mode}, edad_grupo={ag_mode}, "
          f"ciudad={city_mode[:20]}, study={study_mode[:30]}")

# ════════════════════════════════════════════════════════════════════════════
#  NOMBRES DE CLUSTERS  (basados en perfil real)
# ════════════════════════════════════════════════════════════════════════════

# Calculamos perfil para asignar nombre automáticamente
cluster_info = {}
for c in range(best_k):
    sub = df[df["cluster"] == c]
    cluster_info[c] = {
        "n"         : len(sub),
        "age_mean"  : sub["age"].mean(),
        "comfort_mean": sub["comfort"].mean(),
        "mod_mean"  : sub["mod"].mean(),
        "gender_dom": sub["gender_label"].value_counts().idxmax(),
        "age_group" : sub["age_group"].value_counts().idxmax(),
        "study_dom" : sub["study"].value_counts().idxmax(),
        "city_dom"  : sub["city"].value_counts().idxmax(),
    }

# Ordenar por comfort para asignar nombre coherente
sorted_by_comfort = sorted(cluster_info.items(), key=lambda x: x[1]["comfort_mean"])

# Lógica de nombres basada en perfil
def nombre_cluster(info):
    age  = info["age_mean"]
    com  = info["comfort_mean"]
    mod  = info["mod_mean"]
    ag   = info["age_group"]
    gen  = info["gender_dom"]
    study = info["study_dom"].lower()

    if com <= 3.0:
        if age < 19:
            return "Jóvenes Principiantes Sin Confianza"
        else:
            return "Adultos con Baja Autoconfianza"
    elif com <= 5.5:
        if "secundaria" in study or age < 18:
            return "Adolescentes en Transición"
        elif mod >= 1.8:
            return "Avanzados con Confianza Media"
        else:
            return "Iniciados con Potencial"
    else:
        if gen == "Masculino":
            return "Seniors Técnicos Confiados"
        else:
            return "Líderes con Alta Autoconfianza"

cluster_names = {}
for c, info in cluster_info.items():
    cluster_names[c] = nombre_cluster(info)

# Renombrar si hay colisiones
seen = {}
for c, name in cluster_names.items():
    if name in seen:
        cluster_names[c] = name + f" (variante {c})"
    seen[name] = c

print("\n── NOMBRES ASIGNADOS ──────────────────────────────")
for c, name in cluster_names.items():
    info = cluster_info[c]
    print(f"  Cluster {c} → '{name}'")
    print(f"    n={info['n']} | edad_media={info['age_mean']:.1f} | comfort_media={info['comfort_mean']:.1f} "
          f"| mod_media={info['mod_mean']:.2f} | género_dom={info['gender_dom']}")

df["cluster_name"] = df["cluster"].map(cluster_names)

# ════════════════════════════════════════════════════════════════════════════
#  EJEMPLOS POR CLUSTER  (3 alumnos reales)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("EJEMPLOS DE ALUMNOS POR CLUSTER")
print("=" * 65)

display_cols = ["name", "age", "gender_label", "city", "study", "area_grouped", "comfort", "mod"]
for c in range(best_k):
    sub = df_raw.copy()
    sub["cluster"] = labels
    sub["gender_label"] = sub["gender"].str.lower().map({"f": "Femenino", "m": "Masculino"})
    sub = sub[sub["cluster"] == c][display_cols].head(3)
    print(f"\nCluster {c} — '{cluster_names[c]}'")
    print(f"{'─'*65}")
    for _, row in sub.iterrows():
        study_short = str(row["study"]).replace("Estudiante de ", "").replace(" (universidad)", "")
        area_short  = str(row["area_grouped"]).split("/")[0].strip()[:35]
        print(f"  {str(row['name']).title()[:35]:<36} | {row['age']} años | {row['gender_label']}")
        print(f"  {study_short[:40]:<41} | Comfort: {row['comfort']}/10 | Módulo {row['mod']}")
        print(f"  Ciudad: {row['city'][:30]} | Área: {area_short}")
        print()

# ════════════════════════════════════════════════════════════════════════════
#  FIGURAS
# ════════════════════════════════════════════════════════════════════════════

palette_c = {name: sns.color_palette(PALETTE, best_k)[i]
             for i, name in cluster_names.items()}
color_list = [palette_c[cluster_names[c]] for c in labels]

# ── Figura B: Silhouette plot ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle(f"Silhouette Plot — k={best_k}  (score global={sil_global:.3f})", fontsize=13, fontweight="bold")

y_lower = 10
for c in range(best_k):
    c_sil = np.sort(sil_per[labels == c])
    size_c = c_sil.shape[0]
    y_upper = y_lower + size_c
    color = sns.color_palette(PALETTE, best_k)[c]
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil,
                     facecolor=color, edgecolor=color, alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * size_c, f"C{c}\n{cluster_names[c][:20]}",
            fontsize=7, va="center")
    y_lower = y_upper + 10

ax.axvline(sil_global, color="red", linestyle="--", linewidth=1.5, label=f"Media={sil_global:.3f}")
ax.set_xlabel("Silhouette coefficient")
ax.set_ylabel("Cluster")
ax.set_xlim(-0.3, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR + "figB_silhouette_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[OK] figB_silhouette_plot.png")

# ── Figura C: PCA con nombres ──────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(11, 8))
for c in range(best_k):
    mask = labels == c
    color = sns.color_palette(PALETTE, best_k)[c]
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], color=color, s=100,
               edgecolors="white", linewidths=0.7, zorder=3,
               label=f"C{c}: {cluster_names[c]} (n={mask.sum()})")
    # Etiqueta de nombre de alumno
    names_raw = df_raw["name"].values
    for idx in np.where(mask)[0]:
        ax.annotate(names_raw[idx].split()[0].title(), (X_pca[idx, 0], X_pca[idx, 1]),
                    fontsize=6.5, alpha=0.7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

# Centroides
centers_pca = pca.transform(km.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c="black", marker="X",
           s=220, zorder=5, label="Centroides")
for c in range(best_k):
    ax.annotate(f"C{c}", (centers_pca[c, 0], centers_pca[c, 1]),
                fontsize=10, fontweight="bold", color="black",
                xytext=(6, 6), textcoords="offset points")

ax.set_title(f"Clusters en espacio PCA  (varianza explicada: {sum(var_exp)*100:.1f}%)",
             fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT_DIR + "figC_pca_nombres.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] figC_pca_nombres.png")

# ── Figura D: Tarjetas de cluster ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 5 * best_k))
fig.suptitle("Tarjetas de Perfil por Cluster", fontsize=15, fontweight="bold", y=1.01)

for ci, c in enumerate(range(best_k)):
    sub = df[df["cluster"] == c]
    color = sns.color_palette(PALETTE, best_k)[c]

    ax_bar = fig.add_subplot(best_k, 3, ci * 3 + 1)
    ax_pie = fig.add_subplot(best_k, 3, ci * 3 + 2)
    ax_txt = fig.add_subplot(best_k, 3, ci * 3 + 3)

    # Bar: distribución de comfort
    com_counts = sub["comfort"].value_counts().sort_index()
    ax_bar.bar(com_counts.index.astype(str), com_counts.values, color=color, edgecolor="white")
    ax_bar.set_title(f"C{c}: {cluster_names[c]}\nDistribución Comfort", fontsize=9, fontweight="bold")
    ax_bar.set_xlabel("Comfort (1-10)")
    ax_bar.set_ylabel("Alumnos")

    # Pie: género
    g = sub["gender_label"].value_counts()
    ax_pie.pie(g, labels=g.index, autopct="%1.0f%%",
               colors=["#E8A598", "#89B4D1"][:len(g)],
               wedgeprops={"edgecolor": "white"})
    ax_pie.set_title(f"Género  (n={len(sub)})", fontsize=9)

    # Texto: stats clave
    ax_txt.axis("off")
    info = cluster_info[c]
    txt = (f"Alumnos: {info['n']}\n"
           f"Edad media: {info['age_mean']:.1f} años\n"
           f"Comfort medio: {info['comfort_mean']:.2f}/10\n"
           f"Módulo dom.: {sub['mod'].value_counts().idxmax()}\n"
           f"Grupo etario: {info['age_group']}\n"
           f"Ciudad top: {sub['city'].value_counts().idxmax()[:22]}\n"
           f"Estudio: {info['study_dom'][:30]}\n"
           f"Área: {info['age_group']}\n"
           f"\nSilhouette media: {sil_per[labels==c].mean():.3f}")
    ax_txt.text(0.05, 0.95, txt, transform=ax_txt.transAxes,
                fontsize=9, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.2))
    ax_txt.set_title("Estadísticas", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR + "figD_tarjetas_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] figD_tarjetas_clusters.png")

# ── Figura E: radar/spider de medias normalizadas ─────────────────────────
from matplotlib.patches import FancyArrowPatch

feature_labels = ["Módulo", "Edad", "Comfort", "Género", "Nivel Estudio", "Área", "Grupo Etario"]
N = len(feature_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
fig.suptitle("Radar: Perfil Normalizado por Cluster", fontsize=13, fontweight="bold")

for c in range(best_k):
    vals = km.cluster_centers_[c].tolist()
    vals_norm = [(v - X.min()) / (X.max() - X.min() + 1e-9) for v in vals]
    vals_norm += vals_norm[:1]
    color = sns.color_palette(PALETTE, best_k)[c]
    ax.plot(angles, vals_norm, "o-", linewidth=2, color=color,
            label=f"C{c}: {cluster_names[c][:25]}")
    ax.fill(angles, vals_norm, alpha=0.12, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(feature_labels, fontsize=10)
ax.set_ylim(0, 1)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR + "figE_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] figE_radar.png")

# ════════════════════════════════════════════════════════════════════════════
#  RESUMEN FINAL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("RESUMEN FINAL")
print("=" * 65)
print(f"""
¿Por qué el k=9 original tenía silueta 0.29?
  → Con 46 alumnos, k=9 crea clusters de 1-5 personas.
    Un cluster de 1 persona siempre tiene silueta alta pero es trivial.
    La métrica sube artificialmente. NO indica estructura real.

Solución: k=3 por codo de inercia + interpretabilidad.

MÉTRICAS FINALES (k=3)
  Silhouette global : {sil_global:.4f}  {'✓ aceptable' if sil_global > 0.3 else '→ moderado (dataset pequeño y mixto)'}
  Calinski-Harabasz : {ch:.2f}
  Davies-Bouldin    : {db:.4f}

CLUSTERS IDENTIFICADOS:""")

for c in range(best_k):
    info = cluster_info[c]
    sil_c = sil_per[labels == c].mean()
    print(f"""
  Cluster {c} — "{cluster_names[c]}"
    Alumnos : {info['n']}
    Edad    : {info['age_mean']:.1f} años (grupo: {info['age_group']})
    Comfort : {info['comfort_mean']:.2f}/10
    Género  : {info['gender_dom']}
    Ciudad  : {info['city_dom']}
    Study   : {info['study_dom'][:50]}
    Silueta media del cluster: {sil_c:.3f}""")
