import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             recall_score, precision_score,
                             confusion_matrix, precision_recall_curve,
                             average_precision_score)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraude em Cartões — Liga DS",
    page_icon="💳",
    layout="wide"
)

C0   = "#2563EB"
C1   = "#DC2626"
GRAY = "#64748B"

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor":   "none",
    "axes.edgecolor":   "#E2E8F0",
    "axes.grid":        True,
    "grid.color":       "#E2E8F0",
    "grid.linewidth":   0.6,
    "font.size":        11,
})

# ─────────────────────────────────────────────────────────────────
# DADOS E MODELO — com redução de memória
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def carregar_dados():
    caminho = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    for arquivo in os.listdir(caminho):
        if arquivo.endswith(".csv"):
            df = pd.read_csv(os.path.join(caminho, arquivo))
            break

    df = df.drop_duplicates().reset_index(drop=True)

    # ── CORREÇÃO 1: reduz para float32 (metade da memória do float64) ──
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")

    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]]).astype("float32")
    df["Time_scaled"]   = scaler.fit_transform(df[["Time"]]).astype("float32")
    df_modelo = df.drop(["Amount", "Time"], axis=1)

    # ── CORREÇÃO 2: usa 50% dos dados para treino ──────────────────
    # Mantém TODAS as fraudes + amostra das legítimas
    fraudes   = df_modelo[df_modelo["Class"] == 1]
    legitimas = df_modelo[df_modelo["Class"] == 0].sample(
        frac=0.5, random_state=42
    )
    df_modelo_leve = pd.concat([fraudes, legitimas]).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    return df, df_modelo_leve

@st.cache_resource(show_spinner=False)
def treinar(_df_modelo):
    X = _df_modelo.drop("Class", axis=1)
    y = _df_modelo["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── CORREÇÃO 3: SMOTE com sampling_strategy=0.1 ────────────────
    # cria fraudes até elas serem 10% do total — não 50%
    # muito mais leve e ainda resolve o desbalanceamento
    smote = SMOTE(random_state=42, sampling_strategy=0.1)
    X_tr_bal, y_tr_bal = smote.fit_resample(X_train, y_train)

    # ── CORREÇÃO 4: 50 árvores e n_jobs=1 ─────────────────────────
    # n_jobs=-1 usa todos os núcleos — no Cloud gratuito isso consome
    # muita memória. n_jobs=1 é mais lento mas cabe no limite.
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_bal, y_tr_bal)

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,       # limita profundidade das árvores — menos memória
        random_state=42,
        n_jobs=1
    )
    rf.fit(X_tr_bal, y_tr_bal)

    return lr, rf, X_test, y_test

with st.spinner("Carregando dados e treinando modelos... (pode levar 1-2 min)"):
    df, df_modelo = carregar_dados()
    lr, rf, X_test, y_test = treinar(df_modelo)

y_proba_lr = lr.predict_proba(X_test)[:, 1]
y_proba_rf = rf.predict_proba(X_test)[:, 1]
y_pred_lr  = lr.predict(X_test)
y_pred_rf  = rf.predict(X_test)

# ─────────────────────────────────────────────────────────────────
# NAVEGAÇÃO
# ─────────────────────────────────────────────────────────────────
st.sidebar.title("💳 Fraude em Cartões")
st.sidebar.markdown("Liga de Data Science · UNICAMP FCA")
st.sidebar.markdown("---")

capitulos = [
    "01 · O Problema",
    "02 · Os Dados",
    "03 · Ajustando o desbalanceamento",
    "04 · Os Modelos",
    "05 · Simulador ao Vivo",
]
cap = st.sidebar.radio("Capítulos", capitulos)
st.sidebar.markdown("---")
st.sidebar.markdown("**Equipe**  \nGleicy · Matheus · Renan · Victor")

# ─────────────────────────────────────────────────────────────────
# CAPÍTULO 1 — O PROBLEMA
# ─────────────────────────────────────────────────────────────────
if cap == capitulos[0]:
    st.title("01 · O Problema")
    st.markdown(
        "Detectar fraudes é um exercício de equilíbrio. "
        "Nosso objetivo é maximizar a captura de fraudes reais (Recall) estritamente dentro de um limite tolerável de atrito com o cliente (Precisão)."
        "Bloqueios indevidos geram insatisfação e altos custos de suporte."
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Transações analisadas", f"{len(df):,}")
    col2.metric("Fraudes reais", f"{int(df['Class'].sum()):,}")
    col3.metric("Taxa de fraude", f"{df['Class'].mean()*100:.3f}%")

    st.markdown("---")
    st.subheader("O cenário")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "**Dataset:** transações de portadores europeus em setembro de 2013, "
            "coletadas ao longo de dois dias consecutivos.\n\n"
            "**Variáveis:**\n"
            "- `Time` → segundos desde a primeira transação\n"
            "- `Amount` → valor em euros\n"
            "- `V1` a `V28` → componentes PCA (privacidade dos titulares)\n"
            "- `Class` → **0** = legítima · **1** = fraude"
        )
    with col2:
        st.markdown(
            "**Por que isso importa?**\n\n"
            "- Falso Negativo → fraude passa → prejuízo financeiro real\n"
            "- Falso Positivo → compra bloqueada → cliente insatisfeito\n\n"
         )
# ─────────────────────────────────────────────────────────────────
# CAPÍTULO 2 — OS DADOS
# ─────────────────────────────────────────────────────────────────
elif cap == capitulos[1]:
    st.title("02 · Os Dados")
    st.markdown(
        "Antes de qualquer modelo, precisamos entender como as transações "
        "se distribuem no tempo e em valor."
    )
    st.markdown("---")
    st.subheader("Valor das transações ao longo do tempo")

    legitimas = df[df["Class"] == 0].sample(3000, random_state=42)
    fraudes   = df[df["Class"] == 1]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(legitimas["Time"], legitimas["Amount"],
               alpha=0.2, s=4, color=C0, label="Legítima (amostra)")
    ax.scatter(fraudes["Time"], fraudes["Amount"],
               alpha=0.9, s=18, color=C1, label="Fraude", zorder=5)
    ax.set_xlabel("Tempo (segundos)"); ax.set_ylabel("Valor (€)")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição do valor")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.hist(df[df["Class"]==0]["Amount"].clip(0,500),
                 bins=50, alpha=0.6, color=C0, label="Legítima", edgecolor="white")
        ax2.hist(df[df["Class"]==1]["Amount"].clip(0,500),
                 bins=50, alpha=0.9, color=C1, label="Fraude", edgecolor="white")
        ax2.set_xlabel("Valor (€)"); ax2.set_ylabel("Frequência")
        ax2.legend()
        st.pyplot(fig2)
        plt.close()

    with col2:
        st.subheader("Estatísticas por classe")
        stats = df.groupby("Class")["Amount"].agg(["mean","median","max"]).round(2)
        stats.index = ["Legítima", "Fraude"]
        stats.columns = ["Média (€)", "Mediana (€)", "Máximo (€)"]
        st.dataframe(stats, use_container_width=True)
        st.markdown("""
        **Observações:**
        - Fraudes tendem a ter valores medianos **menores**
        - Possível padrão de testes com pequenas quantias
        - Outliers de alto valor existem nas duas classes
        """)
      
    st.markdown("---")
    st.subheader("Em que hora do dia as fraudes acontecem mais?")
 
    # Time está em segundos desde a primeira transação.
    # % 86400 pega o "resto" da divisão por um dia inteiro em segundos,
    # ou seja, traz o valor de volta para dentro de um único dia.
    # Dividir por 3600 converte segundos em horas.
    df_hora = df.copy()
    df_hora["hora"] = (df_hora["Time"] % 86400) / 3600
    df_hora["hora"] = df_hora["hora"].astype(int)
 
    col1= st.columns(1)
 
 
    with col1:
        st.markdown("**Taxa de fraude por hora (%)**")
        # groupby agrupa por hora, .mean() na coluna Class dá a proporção
        # de fraude naquela hora (já que Class só tem 0 e 1)
        taxa_por_hora = df_hora.groupby("hora")["Class"].mean() * 100
 
        fig_h2, ax_h2 = plt.subplots(figsize=(6, 3.5))
        cores = [C1 if v > taxa_por_hora.mean() else C0 for v in taxa_por_hora.values]
        ax_h2.bar(taxa_por_hora.index, taxa_por_hora.values,
                  color=cores, edgecolor="white")
        ax_h2.axhline(taxa_por_hora.mean(), color=GRAY, linestyle="--",
                      linewidth=1, label=f"Média ({taxa_por_hora.mean():.2f}%)")
        ax_h2.set_xlabel("Hora do dia"); ax_h2.set_ylabel("% de fraude")
        ax_h2.legend()
        st.pyplot(fig_h2)
        plt.close()
 
        hora_pico = taxa_por_hora.idxmax()
        st.caption(
            f"🔺 Pico de fraude às **{hora_pico}h**, com "
            f"**{taxa_por_hora.max():.2f}%** de taxa de fraude"
        )

    st.markdown("---")
    st.subheader("Top 10 variáveis correlacionadas com fraude")
    corr = df_modelo.corr(numeric_only=True)["Class"].drop("Class")
    corr_top = corr.sort_values(key=abs, ascending=False).head(10).sort_values()
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    colors = [C1 if v > 0 else C0 for v in corr_top.values]
    corr_top.plot(kind="barh", ax=ax3, color=colors, edgecolor="white")
    ax3.axvline(0, color=GRAY, linewidth=0.8)
    ax3.set_xlabel("Correlação com Class")
    st.pyplot(fig3)
    plt.close()
    st.caption("Vermelho = correlação positiva (aumenta risco) · Azul = negativa")

# ─────────────────────────────────────────────────────────────────
# CAPÍTULO 3 — AJUSTANDO O DESBALANCEAMENTO
# ─────────────────────────────────────────────────────────────────
elif cap == capitulos[2]:
    st.title("03 · Ajustando o desbalanceamento")
    st.markdown(
        "A maior barreira para o projeto "
        "é o **desbalanceamento extremo** dos dados."
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("O problema")
        fig, ax = plt.subplots(figsize=(4, 4))
        contagem = df["Class"].value_counts()
        ax.pie(contagem.values,
               labels=["Legítima\n99.83%", "Fraude\n0.17%"],
               colors=[C0, C1], startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title("Distribuição real")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Por que acurácia não funciona aqui?")
        st.markdown("Imagine um modelo que chuta **legítima** para tudo:")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Acurácia", "99.83%")
        col_b.metric("Recall de Fraude", "0%",
                     delta="-100%", delta_color="inverse")
        col_c.metric("Fraudes detectadas", "0 de 492")

        st.error(
            "99.83% de acurácia deixando **492 fraudes passarem**. "
          
        )

        st.markdown("**A solução: SMOTE**")
        st.markdown(
            "SMOTE cria transações fraudulentas **sintéticas** interpolando "
            "entre fraudes reais, balanceando o dataset de treino."
        )

    st.markdown("---")
    

# ─────────────────────────────────────────────────────────────────
# CAPÍTULO 4 — OS MODELOS
# ─────────────────────────────────────────────────────────────────
elif cap == capitulos[3]:
    st.title("04 · Os Modelos")
    st.markdown(
        "Treinamos dois modelos com SMOTE aplicado corretamente. "
        "Métrica principal: **Recall**. Secundária: **AUC-ROC**."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Regressão Logística**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Recall",   f"{recall_score(y_test, y_pred_lr):.2%}")
        m2.metric("Precisão", f"{precision_score(y_test, y_pred_lr):.2%}")
        m3.metric("AUC-ROC",  f"{roc_auc_score(y_test, y_proba_lr):.4f}")

    with col2:
        st.markdown("**Random Forest ★**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Recall",   f"{recall_score(y_test, y_pred_rf):.2%}")
        m2.metric("Precisão", f"{precision_score(y_test, y_pred_rf):.2%}")
        m3.metric("AUC-ROC",  f"{roc_auc_score(y_test, y_proba_rf):.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matrizes de confusão")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, y_pred, titulo in zip(
            axes,
            [y_pred_lr, y_pred_rf],
            ["Regressão Logística", "Random Forest"]
        ):
            cm = confusion_matrix(y_test, y_pred)
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Legítima","Fraude"])
            ax.set_yticklabels(["Legítima","Fraude"])
            ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
            ax.set_title(titulo)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i,j]:,}",
                            ha="center", va="center",
                            color="white" if cm[i,j] > cm.max()/2 else "black",
                            fontweight="bold", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
        st.caption(
            f"Random Forest → detectou **{tp} fraudes** · "
            f"perdeu **{fn}** · gerou **{fp}** alarmes falsos"
        )

    with col2:
        st.subheader("Curva Precisão × Recall")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for y_proba, nome, cor in [
            (y_proba_lr, "Reg. Logística", C0),
            (y_proba_rf, "Random Forest",  C1)
        ]:
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            ax2.plot(rec, prec, color=cor, label=f"{nome} (AP={ap:.3f})")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precisão")
        ax2.legend()
        st.pyplot(fig2)
        plt.close()
        st.caption("AP maior = modelo melhor")

    st.markdown("---")
    st.subheader("Importância das variáveis — Random Forest")
    importances = pd.Series(
        rf.feature_importances_, index=X_test.columns
    ).sort_values(ascending=True).tail(12)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax3, color=C0, edgecolor="white")
    ax3.set_xlabel("Importância relativa")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ─────────────────────────────────────────────────────────────────
# CAPÍTULO 5 — SIMULADOR
# ─────────────────────────────────────────────────────────────────
elif cap == capitulos[4]:
    st.title("05 · Simulador ao Vivo")
    st.markdown(
        "Ajuste o **limiar de decisão** e veja em tempo real como "
        "Recall, Precisão e fraudes detectadas mudam."
    )
    st.markdown("---")

    threshold = st.slider(
        "Limiar de decisão (menor = mais sensível a fraudes)",
        min_value=0.05, max_value=0.95, value=0.50, step=0.01
    )

    y_pred_sim = (y_proba_rf >= threshold).astype(int)
    rec  = recall_score(y_test, y_pred_sim)
    prec = precision_score(y_test, y_pred_sim)
    det  = int(round(rec * y_test.sum()))
    perd = int(y_test.sum()) - det

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall",             f"{rec:.2%}")
    col2.metric("Precisão",           f"{prec:.2%}")
    col3.metric("Fraudes detectadas", f"{det} / {int(y_test.sum())}")
    col4.metric("Fraudes perdidas",   perd,
                delta=f"-{perd}" if perd > 0 else None,
                delta_color="inverse")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matriz de confusão")
        cm = confusion_matrix(y_test, y_pred_sim)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Legítima","Fraude"])
        ax.set_yticklabels(["Legítima","Fraude"])
        ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}",
                        ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black",
                        fontweight="bold", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Precisão × Recall no limiar atual")
        precs, recs, threshs = precision_recall_curve(y_test, y_proba_rf)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(threshs, precs[:-1], color=C0, label="Precisão")
        ax2.plot(threshs, recs[:-1],  color=C1, label="Recall")
        ax2.axvline(threshold, color=GRAY, linestyle="--",
                    label=f"Limiar ({threshold:.2f})")
        ax2.set_xlabel("Limiar"); ax2.set_ylabel("Score")
        ax2.legend()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")
    st.subheader("Relatório completo")
    st.text(classification_report(
        y_test, y_pred_sim, target_names=["Legítima", "Fraude"]
    ))

    if perd == 0:
        st.success("🎯 Todas as fraudes detectadas com esse limiar!")
    elif perd <= 5:
        st.warning(f"⚠️ {perd} fraude(s) escapando com esse limiar.")
    else:
        st.error(f"🚨 {perd} fraudes não detectadas. Considere reduzir o limiar.")
