import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Functie curatare valori lipsa
def fill_missing_values(df):
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# Functie prelucrare date pentru Capitolul 2
def load_and_prepare_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df_cleaned = fill_missing_values(df.copy())

    df_encoded = df_cleaned.copy()
    df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 1, 'Female': 0})
    df_encoded = pd.get_dummies(df_encoded, columns=['BMI Category', 'Sleep Disorder', 'Occupation'], drop_first=True, dtype=int)
    df_encoded = df_encoded.drop(columns=["Person ID"], errors="ignore")

    numeric_cols = [col for col in df_encoded.columns if col not in ['Gender'] and pd.api.types.is_numeric_dtype(df_encoded[col])]

    scaler_std = StandardScaler()
    df_standardized = df_encoded.copy()
    df_standardized[numeric_cols] = scaler_std.fit_transform(df_standardized[numeric_cols])
    df_standardized[numeric_cols] = df_standardized[numeric_cols].round(2)

    scaler_minmax = MinMaxScaler()
    df_minmax = df_encoded.copy()
    df_minmax[numeric_cols] = scaler_minmax.fit_transform(df_minmax[numeric_cols])
    df_minmax[numeric_cols] = df_minmax[numeric_cols].round(2)

    return df_cleaned, df_encoded, df_standardized, df_minmax, numeric_cols


# Generare boxplot pentru outlieri
def generate_boxplots(df, columns):
    fig, axes = plt.subplots(nrows=len(columns), figsize=(10, 5 * len(columns)))
    if len(columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        sns.boxplot(x=df[col], ax=ax, color='salmon')
        ax.set_title(f"Boxplot - {col}")
    return fig

# Capitolul 3 preprocesare date
def preprocess_ch3(file):
    df = pd.read_csv(file)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna("None")
    df = df.drop(columns=["Person ID"], errors="ignore")
    df = df[df['Blood Pressure'].str.contains('/')]
    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
    df['Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
    df.drop(columns=['Blood Pressure'], inplace=True)
    df = df.dropna()
    df = pd.get_dummies(df, columns=['Occupation', 'BMI Category', 'Sleep Disorder'], drop_first=True, dtype=int)

    # Standardizare doar pe coloane relevante numeric
    exclude_from_scaling = ['Age', 'Gender']
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_from_scaling]

    scaler_std = StandardScaler()
    df_std = df.copy()
    df_std[numeric_cols] = scaler_std.fit_transform(df_std[numeric_cols])
    df_std[numeric_cols] = df_std[numeric_cols].round(2)

    scaler_mm = MinMaxScaler()
    df_mm = df.copy()
    df_mm[numeric_cols] = scaler_mm.fit_transform(df_mm[numeric_cols])
    df_mm[numeric_cols] = df_mm[numeric_cols].round(2)

    return df, df_std, df_mm, numeric_cols

# Aplicatie Streamlit
st.set_page_config(page_title="Giuliana & Cristina", layout="centered")
st.markdown("""
    <style>
    .main {background-color: #fff5f5;}
    h1, h2, h3, h4 {color: #b30000;}
    .stButton>button {
        background-color: #ff4d4d;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        color: white;
    }
    .css-1offfwp {
        background-color: #ffe6e6;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align:center;">Sănătatea somnului și stilul de viață Giuliana & Cristina</h1>', unsafe_allow_html=True)

st.sidebar.title("CAPITOLE")
capitol = st.sidebar.radio("Alege capitolul:", ["Capitolul 1", "Capitolul 2", "Capitolul 3"])

if capitol == "Capitolul 1":
    st.subheader("Capitolul 1")

    file_format = st.radio("Alege formatul fișierului:", ('CSV', 'JSON', 'TXT'))
    uploaded_file = st.file_uploader("Încarcă fișierul de date", type=["csv", "json", "txt"])

    if uploaded_file is not None:
        if file_format == 'CSV':
            df = pd.read_csv(uploaded_file)
        elif file_format == 'JSON':
            df = pd.read_json(uploaded_file)
        elif file_format == 'TXT':
            df = pd.read_csv(uploaded_file, delimiter='	')

        st.success(" Fișierul a fost încărcat cu succes!")
        st.markdown(f"**Format fișier:** `{file_format}`")
        st.markdown(f"**Dimensiunea datasetului:** `{df.shape[0]}` rânduri x `{df.shape[1]}` coloane")

        st.markdown("## Datele din fișier")
        st.dataframe(df)

        df_prelucrat = df.drop(columns=['Person ID'])
        st.markdown("## Statistici descriptive ale datelor procesate")
        st.write(df_prelucrat.describe())
        st.markdown("### Interpretare rezultate")
        st.write("Vârsta (Age): Mediana este 43 ani, ceea ce sugerează un eșantion adult, cu vârste între 27 și 59 ani.")
        st.write("Durata somnului (Sleep Duration): Media este 7.13 ore, aproape de recomandarea generală. Minimul este 5.8 ore, maximul 8.5.")
        st.write("Calitatea somnului (Quality of Sleep): Media de 7.31 (din 10 probabil) indică o calitate destul de bună.")
        st.write("Puls (Heart Rate): Medie de 70.17 bpm, tipică pentru un adult sănătos.")
        st.write("Pași zilnici (Daily Steps): Media este 6816 pași, sub ținta de 10.000 pași/zi.")

        st.markdown("## Durata medie a somnului pe categorii de IMC (Indice Masă Corporală)")
        avg_sleep_bmi = df.groupby('BMI Category')['Sleep Duration'].mean().sort_values()

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(avg_sleep_bmi.index, avg_sleep_bmi.values, color='#ff4d4d', edgecolor='black')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

        ax.set_xlabel("Categorie IMC", fontsize=12)
        ax.set_ylabel("Durata medie a somnului (ore)", fontsize=12)
        ax.set_title("Durata medie a somnului pe categorii de IMC", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)

        st.markdown("## Regresie: Durata Somnului vs Nivelul de Stres")
        X = df[['Stress Level']].values
        y = df['Sleep Duration'].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        intercept = model.intercept_
        coeficient = model.coef_[0]
        r_squared = r2_score(y, y_pred)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.set_style("whitegrid")
        sns.scatterplot(x=df['Stress Level'], y=df['Sleep Duration'], color='#b30000', s=60, ax=ax2, label='Observații reale')
        ax2.plot(df['Stress Level'], y_pred, color='#333333', linewidth=2, label='Linie de regresie')

        ax2.set_title('Regresie liniară: Durata somnului în funcție de nivelul de stres', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Nivelul de stres', fontsize=12)
        ax2.set_ylabel('Durata somnului (ore)', fontsize=12)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        st.pyplot(fig2)

        st.markdown("### Statistici regresie")
        st.write(f"**Intercept (β₀):** `{intercept:.4f}`")
        st.write(f"**Coeficient (β₁):** `{coeficient:.4f}`")
        st.write(f"**R² (coeficient de determinare):** `{r_squared:.4f}`")

        st.markdown("### Interpretare rezultate")
        st.write(f"- Dacă nivelul de stres este `0`, durata estimată a somnului este de **{intercept:.2f} ore**.")
        if coeficient < 0:
            st.write(f"- Fiecare creștere cu 1 unitate a stresului **scade** durata somnului cu **{abs(coeficient):.4f} ore**.")
        else:
            st.write(f"- Fiecare creștere cu 1 unitate a stresului **crește** durata somnului cu **{coeficient:.4f} ore**.")
        st.write(f"- Valoarea R² indică faptul că aproximativ **{r_squared * 100:.2f}%** din variația duratei de somn este explicată de nivelul de stres.")
    else:
        st.warning("Te rugăm să încarci un fișier pentru a continua.")

elif capitol == "Capitolul 2":
    st.subheader("Capitolul 2")
    uploaded_file_2 = st.file_uploader("Încarcă fișierul pentru Capitolul 2", type=["csv"], key="cap2")

    if uploaded_file_2 is not None:
        df_cleaned, df_encoded, df_standardized, df_minmax, numeric_cols = load_and_prepare_data(uploaded_file_2)

        stats = df_standardized[numeric_cols].describe().round(2)
        median_vals = df_standardized[numeric_cols].median().round(2)
        agg_gender = df_standardized.groupby('Gender')[numeric_cols].agg(['mean', 'std', 'min', 'max']).round(2)
        agg_bmi = df_standardized.groupby('BMI Category_Overweight')[numeric_cols].agg(['mean', 'std']).round(2)

        st.markdown("### Date standardizate (StandardScaler)")
        st.dataframe(df_standardized[numeric_cols].head(10))

        st.markdown("### Date normalizate (MinMaxScaler)")
        st.dataframe(df_minmax[numeric_cols].head(10))

        st.markdown("### Statistici descriptive")
        st.dataframe(stats)

        st.markdown("### Valori mediane")
        st.dataframe(median_vals)

        st.markdown("### Agregare după gen")
        st.markdown("**Interpretare:** Tabelul de mai jos prezintă media, abaterea standard, minimul și maximul pentru fiecare variabilă numerică, împărțite pe gen. Genul este codificat ca 0 = Female și 1 = Male. De exemplu, se observă că bărbații tind să aibă o durată ușor mai scurtă a somnului și o calitate mai scăzută a acestuia față de femei.")
        st.dataframe(agg_gender)

        st.markdown("### Agregare după categorie IMC")
        st.markdown("**Interpretare:** Tabelul de mai jos compară media și abaterea standard a variabilelor numerice între persoanele supraponderale (valoare 1) și celelalte (valoare 0). Persoanele supraponderale au, în medie, o calitate a somnului mai scăzută, nivel de stres mai mare și o activitate fizică mai redusă.")
        st.dataframe(agg_bmi)

        st.markdown("### Vizualizare Outliers (Boxplot)")
        st.markdown("**Descriere:** Boxplot-urile sunt grafice care arată distribuția unei variabile și ajută la identificarea valorilor extreme (outlieri). Linia din mijloc reprezintă mediana, iar capetele arată intervalul intercuartil. Punctele în afara acestor limite pot indica anomalii sau observații neobișnuite.")
        selected_vars = st.multiselect("Selectează variabile pentru boxplot", numeric_cols, default=numeric_cols[:3])
        if selected_vars:
            fig_outliers = generate_boxplots(df_standardized, selected_vars)
            st.pyplot(fig_outliers)

        st.markdown("### Histogramă interactivă")
        st.markdown("**Descriere:** Histograma împarte valorile unei variabile în intervale (bin-uri) și afișează frecvența valorilor în fiecare interval. Este utilă pentru a înțelege forma distribuției (simetrică, asimetrică, plată etc.).")
        selected_hist_col = st.selectbox("Alege o coloană numerică pentru histogramă", numeric_cols)
        hist_vals = np.histogram(df_standardized[selected_hist_col].dropna(), bins=10)[0]
        st.bar_chart(hist_vals)
    else:
        st.warning("Te rugăm să încarci un fișier CSV pentru a putea continua analiza în Capitolul 2.")

elif capitol == "Capitolul 3":
    st.subheader("Capitolul 3: Analiza avansată")
    uploaded_file = st.file_uploader("Încarcă fișierul CSV pentru Capitolul 3", type=["csv"], key="cap3")

    if uploaded_file is not None:
        df_ch3, df_std, df_mm, numeric_cols = preprocess_ch3(uploaded_file)

        st.markdown("### Regresie liniară multiplă")
        X = df_std[['Sleep Duration', 'Stress Level', 'Physical Activity Level']]
        y = df_std['Quality of Sleep']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        st.text(model.summary())

        st.markdown("#### Interpretare regresie liniară")
        st.write(
            "- Modelul explică 88.5% din variația calității somnului (R² = 0.885).")
        st.write("- Durata somnului are cel mai mare impact pozitiv asupra calității somnului (+0.40).")
        st.write("- Nivelul de stres afectează negativ și semnificativ calitatea somnului (-0.57).")
        st.write("- Activitatea fizică are un efect pozitiv, dar mai putin semnificativ (+0.09).")

        st.markdown("### Regresie logistică")
        if 'Sleep Disorder_Sleep Apnea' in df_ch3.columns:
            y_log = df_ch3['Sleep Disorder_Sleep Apnea']
        else:
            y_log = df_ch3.iloc[:, -1]  # fallback

        st.markdown("#### Interpretare regresie logistică")
        st.write(
            "- Modelul logistic încearcă să prezică dacă o persoană are apnee în somn în funcție de factori precum stresul, somnul și activitatea fizică.")
        st.write(
            "- Dacă în setul de date sunt mult mai mulți oameni fără apnee decât cu apnee, modelul va avea tendința să prezică mai des clasa majoritară (fără apnee), ceea ce poate reduce eficiența reală a previziunii.")

        X_log = df_std.drop(columns=[col for col in df_std.columns if col.startswith('Sleep Disorder_')], errors='ignore')
        X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        st.text(classification_report(y_test, y_pred))

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        st.markdown("#### Interpretare regresie logistică")
        st.write(f"- Acuratețea modelului este de **{accuracy:.2f}**, ceea ce arată cât de bine a clasificat în general.")
        st.write(f"- Precizia este **{precision:.2f}**: dintre toți cei prezisi cu apnee, acest procent chiar suferă de apnee.")
        st.write(f"- Recall-ul este **{recall:.2f}**: procentul celor cu apnee care au fost corect identificați.")
        st.write(f"- F1 Score este **{f1:.2f}** și arată echilibrul între precizie și recall.")

        if accuracy < 0.75:
            st.warning("Modelul are o performanță modestă. Poate fi îmbunătățit prin ajustarea datelor sau a algoritmului.")
        else:
            st.success("Modelul are o performanță decentă și poate oferi clasificări utile.")


        st.markdown("### Corelogramă variabile standardizate")
        import seaborn as sns

        corr = df_std[numeric_cols + ['Gender']].corr()

        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Corelații Pearson între variabile")
        st.pyplot(fig_corr)

    else:
        st.warning("Te rugăm să încarci fișierul pentru a continua.")

