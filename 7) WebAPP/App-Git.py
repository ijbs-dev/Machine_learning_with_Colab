import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Funcao para carregar o dataset
#@st.cache
def get_data():
    return pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQBbdw_C4TBIOwqvRMIU3r-Yv8vwZcHiCZphxmIictpMmPG23LPyjXdutp16aHDvE-D46GTfSOVfPfD/pub?output=csv")

# Funcao para treinar o modelo
def train_model():
    data = get_data()
    x = data.drop("MEDV", axis=1)
    y = data["MEDV"]
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(x, y)
    return rf_regressor

def main():
    # Criando um dataframe
    data = get_data()

    # Treinando o modelo
    model = train_model()
    data

    # Titulo
    st.title("Data App - Prevendo Valores de Imoveis")

    # Subtitulo
    st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de predição de valores de imóveis em Boston.")

    # Verificando o dataset
    st.subheader("Selecionando apenas um pequeno conjunto de atributos")

    # Atributos para serem exibidos por padrão
    defaultcols = ["RM", "PTRATIO", "LSTAT", "MEDV"]

    # Obtendo a lista de opções
    options = data.columns.tolist()

    # Filtrando os valores padrão que estão presentes na lista de opções
    defaultcols_filtered = [col for col in defaultcols if col in options]

    # Definindo atributos a partir do multiselect
    cols = st.multiselect("Atributos", options, default=defaultcols_filtered)

    # Exibindo os top 10 registros do dataframe
    st.dataframe(data[cols].head(10))

    st.subheader("Distribuição de Imoveis por preço")

    # Definindo a faixa de valores
    faixa_valores = st.slider("Faixa de preço", float(data.MEDV.min()), 150., (10.0, 100.0))

    # Filtrando os dados
    dados = data[data['MEDV'].between(left=faixa_valores[0], right=faixa_valores[1])]

    # Plot a distribuição dos dados
    f = px.histogram(dados, x="MEDV", nbins=100, title="Distribuição de Preços")
    f.update_xaxes(title="MEDV")
    f.update_yaxes(title="Total Imóveis")
    st.plotly_chart(f)

    st.sidebar.subheader("Defina os atributos do imóvel para predição")

    # Mapeando dados do usuário para cada atributo
    crim = st.sidebar.number_input("Taxa de Criminalidade", value=data.CRIM.mean())
    indus = st.sidebar.number_input("Proporção de Hectares de Negócio", value=data.INDUS.mean())
    chas = st.sidebar.selectbox("Faz limite com o rio?", ("Sim", "Não"))

    # Transformando o dado de entrada em valor binário
    chas = 1 if chas == "Sim" else 0

    nox = st.sidebar.number_input("Concentração de Oxido Nitrico", value=data.NOX.mean())

    rm = st.sidebar.number_input("Numero de Quartos", value=1)

    ptratio = st.sidebar.number_input("Índice de alunos para professores", value=data.PTRATIO.mean())

    # Inserindo um botão na tela
    btn_predict = st.sidebar.button("Realizar Predição")

    # Verifica se o botão foi acionado
    if btn_predict:
        result = model.predict([[crim, indus, chas, nox, rm, ptratio]])
        st.sidebar.subheader("O valor previsto para o imóvel é:")
        result = "US $ " + str(round(result[0]*10, 2))
        st.sidebar.write(result)

if __name__ == "__main__":
    main()
