#importações
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples, silhouette_score
import shap
# Configuração de páginas
st.set_page_config(page_title="Análise de Hábitos Estudantis", layout="centered")

# Sidebar para navegação
pagina = st.sidebar.selectbox("Escolha a página", ["Página inicial", "Análise exploratória (EDA)", "Clusterização", "Classificação"])


# Carregar dados
df = pd.read_csv("student_habits_performance.csv") 

#Preencher dados nulos
df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])
df.isnull().sum()

# Renomear colunas para português
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

# Mudar categorias para português
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
    df_fitrado = pd.get_dummies(df, columns=['genero'])

    df_filtrado['trabalho_meio_periodo'] = df_filtrado['trabalho_meio_periodo'].replace({
        'Sim': 1,
        'Não': 0
    })

    df_filtrado['qualidade_dieta'] = df_filtrado['qualidade_dieta'].replace({
        'Boa': 2,
        'Ruim': 0,
        'Regular':1
    })

    df_filtrado['nivel_educacao_parental'] = df_filtrado['nivel_educacao_parental'].replace({
        'Ensino Médio': 0,
        'Bacharelado': 1,
        'Mestrado':2,
    })

    df_filtrado['qualidade_internet'] = df_filtrado['qualidade_internet'].replace({
        'Boa': 2,
        'Ruim': 0,
        'Mediana':1
    })


    df_filtrado['atividades_extracurriculares'] = df_filtrado['atividades_extracurriculares'].replace({
        'Sim': 1,
        'Não': 0
    })

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

#scatterplot notas por tempo em redes sociais
    plt.figure(figsize=(8,5))
    sns.regplot(data=df_filtrado, x='horas_redes_sociais', y='nota_exame', line_kws={'color':'red'})
    plt.title('horas de redes sociais vs Nota Final (com linha de tendência)')
    plt.xlabel('horas de redes sociais')
    plt.ylabel('Nota Final')
    st.pyplot(plt.gcf())

#scatterplot notas por tempo em redes sociais
    plt.figure(figsize=(8,5))
    sns.regplot(data=df_filtrado, x='horas_netflix', y='nota_exame', line_kws={'color':'red'})
    plt.title('horas de streaming vs Nota Final (com linha de tendência)')
    plt.xlabel('horas de streaming')
    plt.ylabel('Nota Final')
    st.pyplot(plt.gcf())



elif pagina == "Clusterização":
    st.title("Clusterização de Hábitos Estudantis")
    st.write("""
    Esta seção permite explorar a clusterização dos hábitos estudantis e seu impacto no desempenho acadêmico.
    """)

    
    # --- Preparar dados para clusterização ---
    df_cluster = df.copy()
    colunas_para_cluster = [
    'idade', 
    'horas_estudo_por_dia', 
    'horas_sono', 
    'nota_exame',
    'frequencia_exercicios_fisicos',
    'avaliacao_saude_mental',
    'frequencia_aulas',
]

# Se tiver LabelEncoder rodado antes, elas já vão estar numéricas
    X = df_cluster[colunas_para_cluster]
    # Padroniza
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Método do cotovelo ---
    st.subheader("Análise do número ideal de clusters (Método do Cotovelo)")
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title("Método do Cotovelo")
    ax.set_xlabel("Número de clusters")
    ax.set_ylabel("Soma das distâncias quadradas intra-clusters (Inertia)")
    ax.grid(True)
    st.pyplot(fig)

    # --- Escolha do número de clusters ---
    n_clusters = st.slider("Escolha o número de clusters", min_value=2, max_value=10, value=3)

    # --- KMeans ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df_result = df.copy()
    df_result['Cluster'] = clusters

    # --- PCA para visualização ---
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = clusters

    st.subheader("Visualização dos clusters via PCA")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', alpha=0.7, s=80, ax=ax)
    ax.set_title("Clusters projetados em 2D pelo PCA")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(title="Cluster")
    st.pyplot(fig)

    # --- Tabela médias numéricas ---
    st.subheader("Médias das variáveis numéricas por cluster")
    numericas = [col for col in df_result.select_dtypes(include=['int64', 'float64']).columns if col != 'Cluster']
    medias_cluster = df_result.groupby('Cluster')[numericas].mean().round(2)
    st.dataframe(medias_cluster)

    # --- Gráficos barras numéricas ---
    st.subheader("Gráficos de barras das variáveis numéricas por cluster")
    for col in numericas:
        fig, ax = plt.subplots(figsize=(6,4))
        medias_cluster[col].plot(kind='bar', ax=ax)
        ax.set_title(f"Média de {col} por Cluster")
        ax.set_ylabel(col)
        ax.set_xlabel("Cluster")
        st.pyplot(fig)

    
# aqui adicionamos a silhueta
    # --- Calcular a silhueta detalhada ---
    sample_silhouette_values = silhouette_samples(X_scaled, clusters)

# Calcula o score médio da silhueta (linha vertical no gráfico)
    score_silhouette = silhouette_score(X_scaled, clusters)

    st.subheader("Silhueta")

    fig, ax = plt.subplots(figsize=(10, 6))

    y_lower = 10  # onde começa a plotagem no eixo y

    for i in range(n_clusters):
    # pega os valores da silhueta do cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.tab10(i)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

    # rotula cluster no meio
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # espaço entre clusters

    ax.set_title("Gráfico da silhueta por cluster")
    ax.set_xlabel("Coeficiente de silhueta")
    ax.set_ylabel("Pontos amostrados")
    ax.axvline(x=score_silhouette, color="red", linestyle="--")
    ax.set_yticks([])  # esconde y ticks
    ax.set_xlim([-0.1, 1])
    st.pyplot(fig)
#Página de Classificação
elif pagina == "Classificação":
    st.title("Predição de Desempenho Acadêmico")
    st.write("""
    Esta seção permite explorar como os hábitos estudantis podem ser usados para prever resultados acadêmicos.
    """)

    # Encoding categóricos
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=['genero'])

    df_encoded['trabalho_meio_periodo'] = df_encoded['trabalho_meio_periodo'].replace({'Sim':1, 'Não':0})
    df_encoded['qualidade_dieta'] = df_encoded['qualidade_dieta'].replace({'Boa':2, 'Regular':1, 'Ruim':0})
    df_encoded['nivel_educacao_parental'] = df_encoded['nivel_educacao_parental'].replace({'Ensino Médio':0, 'Bacharelado':1, 'Mestrado':2})
    df_encoded['qualidade_internet'] = df_encoded['qualidade_internet'].replace({'Boa':2, 'Mediana':1, 'Ruim':0})
    df_encoded['atividades_extracurriculares'] = df_encoded['atividades_extracurriculares'].replace({'Sim':1, 'Não':0})

    X = df_encoded.drop(['id_aluno', 'nota_exame'], axis=1)
    y = (df_encoded['nota_exame'] >= 70).astype(int)  # Passou/Não passou

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.subheader("Divisão dos dados")

    st.write(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
    st.write(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")

    st.write("Distribuição das classes no treino:")
    st.write(y_train.value_counts())

    st.write("Distribuição das classes no teste:")
    st.write(y_test.value_counts())


    # Padronização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    modelos_preditivos = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(class_weight='balanced'),
        "SVM": SVC(probability=True, class_weight='balanced'),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    melhores_modelos = {}
    for nome, modelo in modelos_preditivos.items():
        modelo.fit(X_train, y_train)
        preds = modelo.predict(X_test)
        acc = accuracy_score(y_test, preds)*100
        cm = confusion_matrix(y_test, preds)
        relatorio = classification_report(y_test, preds, target_names=["Reprovação", "Aprovação"])

        st.subheader(f"{nome}")
        st.text(f"Acurácia: {acc:.2f}%")
        st.text(relatorio)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Reprovação", "Aprovação"])
        disp.plot(cmap='Blues')
        plt.title(f'Matriz de Confusão - {nome}')
        st.pyplot(plt.gcf())
        plt.clf()

        melhores_modelos[nome] = (modelo, acc)

    # Escolher melhor modelo para explicabilidade SHAP
    melhor_nome = max(melhores_modelos, key=lambda k: melhores_modelos[k][1])
    melhor_modelo = melhores_modelos[melhor_nome][0]

    st.header(f"Explicabilidade do modelo final: {melhor_nome} (SHAP)")

    # SHAP
    explainer = shap.Explainer(melhor_modelo, X_train)
    shap_values = explainer(X_test)

    #st.set_option('deprecation.showPyplotGlobalUse', False)
    ig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, features=X_test, feature_names=X.columns)
    st.pyplot(plt.gcf())