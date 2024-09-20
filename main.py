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

# Agrupar por nome da peça e somar as quantidades
pecas_agrupadas = pecas_df.groupby('nome_da_peca')['quantidade'].sum().reset_index()

# Criar o gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(pecas_agrupadas['nome_da_peca'], pecas_agrupadas['quantidade'], color='skyblue')
plt.title('Quantidade de Peças Mais Trocadas no Período')
plt.xlabel('Nome da Peça')
plt.ylabel('Quantidade Trocada')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Mostrar o gráfico
plt.show()
