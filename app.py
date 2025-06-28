import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Configuração inicial
st.set_page_config(page_title="Análise de Hábitos Estudantis", layout="centered")

# Sidebar para navegação
pagina = st.sidebar.selectbox("Escolha a página", ["Página inicial", "Análise exploratória (EDA), Clusterização"])


# Carregar dados
df = pd.read_csv("student_habits_performance.csv") 

df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])
df.isnull().sum()

categoricas = df.select_dtypes(include='object').columns
for coluna in categoricas:
    print(coluna)

df=df.rename(columns={
    'student_id': 'id_aluno',
    'age': 'idade',
    'gender': 'genero',
    'study_hours_per_day': 'horas_estudo_por_dia',
    'social_media_hours': 'horas_redes_sociais',
    'netflix_hours': 'horas_netflix',
    'part_time_job':'trabalho_meio_periodo',
    'attendance_percentage': 'frequencia_aulas',
    'sleep_hours': 'horas_sono',
    'diet_quality':'qualidade_dieta',
    'exercise_frequency': 'frequencia_exercicios_fisicos',
    'parental_education_level': 'nivel_educacao_parental',
    'internet_quality':'qualidade_internet',
    'mental_health_rating':'avaliacao_saude_mental',
    'extracurricular_participation': 'atividades_extracurriculares',
    'exam_score': 'nota_exame'

})

# Mapeamento de categorias para português
df['genero'] = df['genero'].replace({
    'Male': 'Maculino',
    'Female': 'Feminino',
    'Other': 'Outro'
})

df['trabalho_meio_periodo'] = df['trabalho_meio_periodo'].replace({
    'Yes': 'Sim',
    'No': 'Não'
})

df['qualidade_dieta'] = df['qualidade_dieta'].replace({
    'Good': 'Boa',
    'Poor': 'Ruim',
    'Fair':'Regular'
})
df['nivel_educacao_parental'] = df['nivel_educacao_parental'].replace({
    'High School': 'Ensino Médio',
    'Bachelor': 'Bacharelado',
    'Master':'Mestrado',
})
df['qualidade_internet'] = df['qualidade_internet'].replace({
    'Good': 'Boa',
    'Poor': 'Ruim',
    'Average':'Mediana'
})

df['atividades_extracurriculares'] = df['atividades_extracurriculares'].replace({
    'Yes': 'Sim',
    'No': 'Não'
})


# Página inicial
if pagina == "Página inicial":
    st.title("Bem-vindo ao Dashboard de Hábitos Estudantis")
    st.write("""
    Este app permite explorar como hábitos estudantis, saúde mental e exercícios físicos influenciam
    o desempenho acadêmico dos estudantes.
    
    - Use o menu ao lado para navegar para a análise exploratória.
    - Veja gráficos interativos, correlações e muito mais.
    """)
    st.write(df.head())

# Página de EDA
elif pagina == "Análise exploratória (EDA)":
   
    st.sidebar.header("Filtros")

# Filtro de gênero (multiselect, com tudo selecionado por padrão)
    generos = df['genero'].unique().tolist()
    filtro_genero = st.sidebar.multiselect("Gênero", options=generos, default=generos)
# Filtro de trabalho (multiselect, com tudo selecionado por padrão)
    trabalho = df['trabalho_meio_periodo'].unique().tolist()
    filtro_trabalho = st.sidebar.multiselect("Trabalho de meio período", options=trabalho, default=trabalho)
# Filtro de internet (multiselect, com tudo selecionado por padrão)
    internet = df['qualidade_internet'].unique().tolist()
    filtro_internet = st.sidebar.multiselect("Qualidade de internet", options=internet, default=internet)
# Filtro de dieta (multiselect, com tudo selecionado por padrão)
    dieta = df['qualidade_dieta'].unique().tolist()
    filtro_dieta = st.sidebar.multiselect("Qualidade de dieta", options=dieta, default=dieta)
# Filtro de extracurriculares (multiselect, com tudo selecionado por padrão)
    extra = df['atividades_extracurriculares'].unique().tolist()
    filtro_extra = st.sidebar.multiselect("Atvidades Extracurrículares", options=extra, default=extra)
# Filtro educação (multiselect, com tudo selecionado por padrão)
    educacao = df['nivel_educacao_parental'].unique().tolist()
    filtro_educacao = st.sidebar.multiselect("Nível de educação parental", options=educacao, default=educacao)

# Filtro de idade (slider com mínimo e máximo do dataset)
    idade_min = int(df['idade'].min())
    idade_max = int(df['idade'].max())
    filtro_idade = st.sidebar.slider("Idade", min_value=idade_min, max_value=idade_max, value=(idade_min, idade_max))
# Filtro de horas de estudo (slider com mínimo e máximo do dataset)
    estudo_min = int(df['horas_estudo_por_dia'].min())
    estudo_max = int(df['horas_estudo_por_dia'].max())
    filtro_estudo = st.sidebar.slider("Média de horas de estudo por dia", min_value=estudo_min, max_value=estudo_max, value=(estudo_min, estudo_max))
# Filtro de horas de redes sociais(slider com mínimo e máximo do dataset)
    social_min = int(df['horas_redes_sociais'].min())
    social_max = int(df['horas_redes_sociais'].max())
    filtro_social= st.sidebar.slider("Média de horas de em redes sociais", min_value=social_min, max_value=social_max, value=(social_min, social_max))

# Filtro frequência exercícios (slider)
    ex_min = int(df['frequencia_exercicios_fisicos'].min())
    ex_max = int(df['frequencia_exercicios_fisicos'].max())
    filtro_exercicios = st.sidebar.slider("Frequência Exercícios Físicos", min_value=ex_min, max_value=ex_max, value=(ex_min, ex_max))
# Filtro nota (slider)
    nota_min = int(df['nota_exame'].min())
    nota_max = int(df['nota_exame'].max())
    filtro_nota = st.sidebar.slider("Nota no exame", min_value=nota_min, max_value=nota_max, value=(nota_min, nota_max))
# Filtro saude mental (slider)
    saude_min = int(df['avaliacao_saude_mental'].min())
    saude_max = int(df['avaliacao_saude_mental'].max())
    filtro_saude = st.sidebar.slider("Avaliação de saúde mental", min_value=saude_min, max_value=saude_max, value=(saude_min, saude_max))
# Filtro sono (slider)
    sono_min = int(df['horas_sono'].min())
    sono_max = int(df['horas_sono'].max())
    filtro_sono = st.sidebar.slider("Horas de sono", min_value=sono_min, max_value=sono_max, value=(sono_min, sono_max))
# Filtro Frequencia (slider)
    aula_min = float(df['frequencia_aulas'].min())
    aula_max = float(df['frequencia_aulas'].max())
    filtro_aula = st.sidebar.slider("Frequência em aulas", min_value=aula_min, max_value=aula_max, value=(aula_min, aula_max))

# Filtrando o dataframe baseado nos filtros selecionados
    df_filtrado = df[
        (df['genero'].isin(filtro_genero)) &
        (df['trabalho_meio_periodo'].isin(filtro_trabalho)) &
        (df['qualidade_internet'].isin(filtro_internet)) &
        (df['qualidade_dieta'].isin(filtro_dieta)) &
        (df['atividades_extracurriculares'].isin(filtro_extra)) &
        (df['nivel_educacao_parental'].isin(filtro_educacao)) &
        (df['idade'] >= filtro_idade[0]) & (df['idade'] <= filtro_idade[1]) &
        (df['horas_estudo_por_dia'] >= filtro_estudo[0]) & (df['horas_estudo_por_dia'] <= filtro_estudo[1]) &
        (df['horas_redes_sociais'] >= filtro_social[0]) & (df['horas_redes_sociais'] <= filtro_social[1]) &
        (df['frequencia_exercicios_fisicos'] >= filtro_exercicios[0]) & (df['frequencia_exercicios_fisicos'] <= filtro_exercicios[1]) &
        (df['avaliacao_saude_mental'] >= filtro_saude[0]) & (df['avaliacao_saude_mental'] <= filtro_saude[1]) &
        (df['horas_sono'] >= filtro_sono[0]) & (df['horas_sono'] <= filtro_sono[1]) &
        (df['frequencia_aulas'] >= filtro_aula[0]) & (df['frequencia_aulas'] <= filtro_aula[1]) &
        (df['nota_exame'] >= filtro_nota[0]) & (df['nota_exame'] <= filtro_nota[1])
    ]

#Pagina eda
    st.title("Análise Exploratória de Dados")

    st.subheader("Variáveis")

    df_no_id = df_filtrado.drop('id_aluno', axis=1)

    def plot_coluna(df, coluna):
        fig, ax = plt.subplots(figsize=(7, 4))
    
        if df[coluna].dtype in ['int64', 'float64']:
            sns.histplot(data=df, x=coluna, kde=True, ax=ax)
            ax.set_title(f'Distribuição de {coluna}')
        else:
            sns.countplot(data=df, x=coluna, ax=ax)
            ax.set_title(f'Contagem de {coluna}')
            ax.tick_params(axis='x', rotation=0)
    
        st.pyplot(fig)
        plt.close(fig)
    
        df_no_id = df_filtrado.drop('id_aluno', axis=1)

    
    df_no_id = df_filtrado.drop('id_aluno', axis=1)

    for col in df_no_id.columns:
        plot_coluna(df_no_id, col)
# prompt: grafico de correlação das variaveis

# Selecionar apenas as colunas numéricas para o cálculo da correlação
    df_numeric = df_filtrado.select_dtypes(include=np.number)

    correlation_matrix = df_numeric.corr()

    st.subheader("Relações entre variáveis")
#heatmap da matriz de correlação
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
    plt.title('Matriz de Correlação das Variáveis Numéricas')
    st.pyplot(plt.gcf())

#scatterplot horas de estudo e nota do exame
    plt.figure(figsize=(8,5))
    sns.regplot(data=df_filtrado, x='horas_estudo_por_dia', y='nota_exame', scatter=True, line_kws={"color":"red"})
    plt.title("Horas de estudo vs Nota")
    plt.xlabel("Horas de estudo")
    plt.ylabel("Nota final")
    st.pyplot(plt.gcf())

#boxplot notas por saude mental
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df_filtrado, x='avaliacao_saude_mental', y='nota_exame')
    plt.title('Notas por Saúde Mental')
    plt.xlabel('Saúde Mental')
    plt.ylabel('Nota Final')
    st.pyplot(plt.gcf())

#scatterplot notas por saude mental
    plt.figure(figsize=(8,5))
    sns.regplot(data=df_filtrado, x='avaliacao_saude_mental', y='nota_exame', line_kws={'color':'red'})
    plt.title('Saúde Mental vs Nota Final (com linha de tendência)')
    plt.xlabel('Saúde Mental')
    plt.ylabel('Nota Final')
    st.pyplot(plt.gcf())

#boxplot notas por exercicio fisico
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df_filtrado, x='frequencia_exercicios_fisicos', y='nota_exame')
    plt.title('Notas por Saúde Mental')
    plt.xlabel('Saúde Mental')
    plt.ylabel('Nota Final')
    st.pyplot(plt.gcf())

#scatterplot notas por exercicio fisico
    plt.figure(figsize=(8,5))
    sns.regplot(data=df_filtrado, x='frequencia_exercicios_fisicos', y='nota_exame', line_kws={'color':'red'})
    plt.title('exercicio fisico vs Nota Final (com linha de tendência)')
    plt.xlabel('exercicio fisico')
    plt.ylabel('Nota Final')
    st.pyplot(plt.gcf())

#scatterplot notas por horas de sono
    plt.figure(figsize=(8,5))
    sns.regplot(data=df_filtrado, x='horas_sono', y='nota_exame', line_kws={'color':'red'})
    plt.title('horas de sono vs Nota Final (com linha de tendência)')
    plt.xlabel('horas de sono')
    plt.ylabel('nota final')
    st.pyplot(plt.gcf())


elif pagina == "Clusterização":
    st.title("Clusterização de Hábitos Estudantis")
    st.write("""
    Esta seção permite explorar a clusterização dos hábitos estudantis e seu impacto no desempenho acadêmico.
    
    - Use o menu ao lado para navegar para a análise exploratória.
    - Veja gráficos interativos, correlações e muito mais.
    """)
    
    # Aqui você pode adicionar a lógica de clusterização e visualização
    # Exemplo: KMeans, DBSCAN, etc.
    
    st.write("Clusterização ainda não implementada.")
 