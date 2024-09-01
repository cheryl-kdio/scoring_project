# Initialize the project :
print("================================Start================================")
import warnings
warnings.filterwarnings("ignore")

# Imports de fichier

import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading database\n")
file_path = 'base_finale_cible_1.sas7bdat'
try:
    df = pd.read_sas(file_path, format='sas7bdat')
except Exception as e:
    print(f"Erreur lors de la lecture du fichier SAS : {e}")

df.drop(columns=["cible_2","cible_3"],inplace=True)
# To get global information of data.
#df.info()
# 185 columns in total : 153 of types float (~supposed continuous) and 32 of types objects.

#Target is a variable of type Object, transform to int.
print("Transforming type of target column\n")

cible = "cible_1"
df[cible]=df[cible].astype(int)

# Décoder les colonnes encodées en bytes (type objet)

print("Decoding bytes to utf-8\n")
def decode_bytes(val):
    if isinstance(val, bytes):  # Vérifie si la valeur est de type byte
        try:
            return val.decode('utf-8')  # Tente de décoder en 'utf-8'
        except UnicodeDecodeError:
            return val.decode('utf-8', errors='ignore')  # Ignore les erreurs de décodage
    return val

# Appliquer cette fonction à toutes les colonnes de type object
for col in df.select_dtypes(include=[object]):
    df[col] = df[col].map(decode_bytes)

print("Transforming type of date column\n")
df['date'] = pd.to_datetime(df['date'], format = '%Y%m')

print("Split df to test and training samples\n")
from sklearn.model_selection import train_test_split

# 'Stratify' parameter makes a split so that the proportion of values in
# the sample produced will be the same as the proportion of values provided

# Might be interesting to add "sector" etc. 
df['stratify_param'] = df[cible].astype(str) + '_' + df['date'].astype(str)

# Split train-test stratifié en utilisant la colonne combinée
X_train, X_test, y_train, y_test = train_test_split(
    df,  # Les features incluant la date si elle est utilisée comme feature
    df[cible],                 # La variable cible
    stratify=df['stratify_param'],  # La nouvelle colonne pour la stratification
    test_size=0.3,                # La proportion de test
    random_state=42               # set.seed()
)

# Vous pouvez ensuite supprimer la colonne 'stratify_col' si elle n'est pas nécessaire pour l'entraînement
X_train = X_train.drop(['stratify_param'], axis=1)
X_test = X_test.drop(['stratify_param'], axis=1)

print("Identifying categorical and quantitative variables\n")
object_columns = X_train.select_dtypes(include=['object']).columns.tolist()
float_columns = X_train.select_dtypes(include=['float64']).columns.tolist()

print("================================End================================")