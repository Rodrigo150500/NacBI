import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Carregar o dataset JSON
df = pd.read_json('dataset.json')

# Ordenar por data
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values(by='data')

# Converter datas para números (dias desde o início)
df['dia'] = (df['data'] - df['data'].min()).dt.days


# Função para calcular tendência usando regressão linear
def calculate_trend(df, column_name):
    X = df[['dia']].values
    y = df[column_name].values
    model = LinearRegression()
    model.fit(X, y)

    df[f'{column_name}_trend'] = model.predict(X)
    return model.coef_[0], model.intercept_


# Lista de propriedades a analisar
properties = [
    'gasto_de_energia_kWh',
    'tempo_de_operacao_horas',
    'tempo_de_inatividade_horas',
    'tempo_medio_para_trocar_um_componente_minutos',
    'temperatura_de_operacao_C'
]

# Criar um gráfico para cada propriedade
for prop in properties:
    if prop in df.columns:
        coef, intercept = calculate_trend(df, prop)
        plt.figure(figsize=(10, 6))
        plt.plot(df['data'], df[prop], label=f'{prop} Observado')
        plt.plot(df['data'], df[f'{prop}_trend'], label=f'{prop} Tendência', linestyle='--')
        plt.title(f'Tendência de {prop}')
        plt.xlabel('Data')
        plt.ylabel(prop)
        plt.xticks(rotation=45)
        plt.xticks(np.arange(min(df['data']), max(df['data']) + pd.Timedelta(days=1),
                             pd.Timedelta(days=1)), rotation=45)  # Mostrar mais dias no eixo x
        plt.legend()
        plt.grid(True)
        plt.show()

# Transformar a coluna de peças mais trocadas em um formato adequado
pecas_df = df.explode('pecas_mais_trocadas')
pecas_df['nome_da_peca'] = pecas_df['pecas_mais_trocadas'].apply(lambda x: x['nome_da_peca'])
pecas_df['quantidade'] = pecas_df['pecas_mais_trocadas'].apply(lambda x: x['quantidade'])

# Lista de peças únicas
pecas_unicas = pecas_df['nome_da_peca'].unique()


# Função para calcular tendência usando regressão linear para as peças
def calculate_trend_pecas(pecas_df, peca):
    peca_data = pecas_df[pecas_df['nome_da_peca'] == peca].copy()  # Cópia explícita para evitar o erro
    dias = peca_data['dia'].values.reshape(-1, 1)
    quantidades = peca_data['quantidade'].values
    model = LinearRegression()
    model.fit(dias, quantidades)
    peca_data.loc[:, 'tendencia'] = model.predict(dias)
    return peca_data


# Gerar gráficos de linha individuais para cada peça
for peca in pecas_unicas:
    peca_data = calculate_trend_pecas(pecas_df, peca)

    plt.figure(figsize=(10, 6))
    plt.plot(peca_data['data'], peca_data['quantidade'], label='Quantidade Trocada', marker='o')
    plt.plot(peca_data['data'], peca_data['tendencia'], label='Tendência', linestyle='--')

    plt.title(f'Tendência de Trocas para {peca}')
    plt.xlabel('Data')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45)
    plt.xticks(np.arange(min(peca_data['data']), max(peca_data['data']) + pd.Timedelta(days=1),
                         pd.Timedelta(days=1)), rotation=45)  # Mostrar mais dias no eixo x
    plt.legend()
    plt.grid(True)
    plt.show()
