import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, recall_score
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Fraude em Cartões", page_icon="💳", layout="wide")
st.title("💳 Detecção de Fraude em Cartões de Crédito")
st.markdown("**Liga de Data Science — Mini Projeto** · Gleicy, Matheus, Renan e Victor")

# ─────────────────────────────────────────────────────────────────
# CARREGAMENTO DOS DADOS VIA KAGGLEHUB
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    # kagglehub.dataset_download baixa o dataset direto do Kaggle
    # e retorna o caminho da pasta onde os arquivos ficaram
    caminho = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

    # procura o arquivo .csv dentro da pasta baixada
    for arquivo in os.listdir(caminho):
        if arquivo.endswith(".csv"):
            df = pd.read_csv(os.path.join(caminho, arquivo))
            break

    df = df.drop_duplicates().reset_index(drop=True)

    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
    df_modelo = df.drop(["Amount", "Time"], axis=1)

    return df, df_modelo

@st.cache_resource
def treinar_modelo(df_modelo):
    X = df_modelo.drop("Class", axis=1)
    y = df_modelo["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE só no treino — regra de ouro do projeto
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_bal, y_train_bal)

    return rf, X_test, y_test

# barra de progresso visível enquanto carrega e treina
with st.spinner("Carregando dados e treinando modelo... (pode levar alguns minutos)"):
    df, df_modelo = carregar_dados()
    rf, X_test, y_test = treinar_modelo(df_modelo)

# ─────────────────────────────────────────────────────────────────
# ABAS DO APP
# ─────────────────────────────────────────────────────────────────
aba1, aba2, aba3 = st.tabs(["📊 Visão Geral", "📈 Análise", "🤖 Modelo"])

# ── ABA 1 — VISÃO GERAL ──────────────────────────────────────────
with aba1:
    st.subheader("Resumo do Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Transações", f"{len(df):,}")
    col2.metric("Fraudes", f"{int(df['Class'].sum()):,}")
    col3.metric("Taxa de Fraude", f"{df['Class'].mean()*100:.3f}%")

    fig, ax = plt.subplots(figsize=(6, 3))
    contagem = df["Class"].value_counts()
    ax.bar(["Legítima", "Fraude"], contagem.values,
           color=["#2563EB", "#DC2626"], edgecolor="white")
    ax.set_title("Distribuição do Alvo")
    st.pyplot(fig)

    st.markdown("""
    **Sobre o dataset:** transações realizadas por cartões de crédito europeus
    em setembro de 2013. As variáveis V1-V28 foram transformadas por PCA
    para preservar a privacidade dos titulares.
    """)

# ── ABA 2 — ANÁLISE ──────────────────────────────────────────────
with aba2:
    st.subheader("Valor das Transações ao Longo do Tempo")

    legitimas = df[df["Class"] == 0].sample(3000, random_state=42)
    fraudes   = df[df["Class"] == 1]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.scatter(legitimas["Time"], legitimas["Amount"],
                alpha=0.3, s=5, color="#2563EB", label="Legítima")
    ax2.scatter(fraudes["Time"], fraudes["Amount"],
                alpha=0.8, s=15, color="#DC2626", label="Fraude")
    ax2.set_xlabel("Tempo (segundos)")
    ax2.set_ylabel("Valor (€)")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Valor médio por classe")
    col1, col2 = st.columns(2)
    col1.metric("Legítima", f"€{df[df['Class']==0]['Amount'].mean():.2f}")
    col2.metric("Fraude",   f"€{df[df['Class']==1]['Amount'].mean():.2f}")

    st.subheader("Top variáveis correlacionadas com fraude")
    corr = df_modelo.corr(numeric_only=True)["Class"].drop("Class")
    corr = corr.sort_values(key=abs, ascending=False).head(10)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    corr.sort_values().plot(kind="barh", ax=ax3, color="#2563EB")
    ax3.set_title("Correlação com Fraude (top 10)")
    st.pyplot(fig3)

# ── ABA 3 — MODELO ───────────────────────────────────────────────
with aba3:
    st.subheader("Simulador de Limiar de Decisão")
    st.markdown(
        "Ajuste o limiar abaixo e veja como o Recall e a Precisão mudam "
        "em tempo real. Limiares menores capturam mais fraudes, mas geram "
        "mais alarmes falsos."
    )

    threshold = st.slider("Limiar de decisão", 0.05, 0.95, 0.50, 0.01)

    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    col1, col2, col3 = st.columns(3)
    col1.metric("Recall",  f"{recall_score(y_test, y_pred):.2%}")
    col2.metric("AUC-ROC", f"{roc_auc_score(y_test, y_proba):.4f}")

    fraudes_detectadas = int(round(recall_score(y_test, y_pred) * y_test.sum()))
    fraudes_perdidas   = int(y_test.sum()) - fraudes_detectadas
    col3.metric("Fraudes perdidas", fraudes_perdidas)

    st.text("Relatório completo de classificação:")
    st.text(classification_report(y_test, y_pred,
            target_names=["Legítima", "Fraude"]))

    st.subheader("Importância das variáveis")
    importances = pd.Series(rf.feature_importances_, index=X_test.columns)
    importances = importances.sort_values(ascending=True).tail(10)
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax4, color="#2563EB")
    ax4.set_title("Top 10 Variáveis — Random Forest")
    st.pyplot(fig4)
