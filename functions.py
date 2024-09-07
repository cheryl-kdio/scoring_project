# Kruska wallis tests

from scipy.stats import kruskal
import pandas as pd
#check kruskal wallis handling of NA
def kruskal_wallis(df, continuous_var,target_name):
    kruskal_result = []
    for col in continuous_var :
        df_clean = df[[col,target_name]].dropna(axis=0)
        group=[group for _, group in df_clean.groupby(target_name)[col]]
        statistic, pvalue = kruskal(*group)
        kruskal_result.append([col,statistic,pvalue])
    result_df = pd.DataFrame(kruskal_result,columns=["Columns","Stat","Pvalue"])
    return result_df

import numpy as np

#Pour discrétiser une variable continue Weighted of Evidence
def iv_woe(data,target,bins=5,show_woe=False,epsilon=1e-16):
    newDF,woeDF = pd.DataFrame(),pd.DataFrame()
    cols=data.columns

    #Run WOE and IV on all independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars],bins,duplicates="drop")
            d0=pd.DataFrame({'x':binned_x,'y':data[target]})
        else:
            d0=pd.DataFrame({'x':data[ivars],'y':data[target]})

        #calculate the nb of events in each group (bin)

        d=d0.groupby("x",as_index=False).agg({"y":["count", "sum"]})
        d.columns = ["Cutoff","N","Events"]

        #calculate % of events in each group
        d['% of Events']=np.maximum(d['Events'],epsilon)/(d['Events'].sum()+epsilon)

        #calculate the non events in each group
        d['Non-Events']=d['N'] - d['Events']
        #calculate % of non-events in each group
        d['% of Non-Events']=np.maximum(d['Non-Events'],epsilon)/(d['Non-Events'].sum()+epsilon)

        #calculate WOE by taking natural log of division of % of non-events and % of events
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE']*(d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0,column="Variable",value=ivars)
        print("--------------------------------------------\n")
        print("Information value of variable " + ivars + " is " + str(round(d["IV"].sum(),6)))
        temp=pd.DataFrame({"Variable":[ivars],"IV":[d["IV"].sum()]},columns=["Variable","IV"])
        newDF=pd.concat([newDF,temp],axis=0)
        woeDF=pd.concat([woeDF,d],axis=0)

        #show woe table
        if show_woe==True:
            print(d)
    return newDF,woeDF

def discretize_with_iv_woe(X_train, cible,date, numerical_columns, bins=5, epsilon=1e-16):
    discretized_data = X_train[[date,cible]].copy()
    discretized_columns = []
    non_discretized_columns = []

    for col in numerical_columns:
        # Appliquer la fonction iv_woe pour obtenir les points de coupure
        result = iv_woe(X_train[[col] + [cible]], cible, bins=bins, show_woe=False, epsilon=epsilon)

        if result[1]["IV"].sum() != 0:  # Si l'IV n'est pas nul, discrétiser
            # Extraire les cutoffs (intervalles)
            cutoffs = result[1]["Cutoff"].unique()
            
            # Si les cutoffs sont des intervalles, extraire les bornes
            if isinstance(cutoffs[0], pd.Interval):
                bins_edges = sorted(set([interval.left for interval in cutoffs] + [interval.right for interval in cutoffs]))
            else:
                # Sinon, traiter les cutoffs comme des valeurs discrètes (par exemple pour des variables catégoriques)
                bins_edges = sorted(cutoffs)
            
            # Discrétiser la colonne en utilisant les bornes et ajouter la colonne discrétisée avec suffixe "_cut"
            discretized_data[col + "_dis"] = pd.cut(X_train[col].copy(), bins=bins_edges, include_lowest=True, duplicates='drop')
            discretized_columns.append(col + "_dis")

            print(f"Discrétisation de la colonne {col} avec les bornes: {bins_edges}")
        else:
            discretized_data[col ] = X_train[col].copy()
            non_discretized_columns.append(col)

    return discretized_data, discretized_columns, non_discretized_columns

import matplotlib.pyplot as plt
def tx_rsq_par_var(df,categ_var,date,target):
    df_times_series=(df.groupby([date,categ_var])[target].mean()*100).reset_index()
    df_pivot=df_times_series.pivot(index=date,columns=categ_var,values=target)
    plt.figure(figsize=(10,6))
    for category in df_pivot.columns:
        plt.plot(df_pivot.index,df_pivot[category],label=category)
        plt.title(f"{categ_var}")
    plt.xlabel("Date")
    plt.ylabel("Tx d'événement")
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()


import seaborn as sns
def combined_barplot_lineplot(df,cat_col,cible):
    if pd.api.types.is_categorical_dtype(df[cat_col]):
        df[cat_col] = df[cat_col].astype(str)
    # Calcul du taux de risque
    tx_rsq = (df.groupby([cat_col])[cible].mean() * 100).reset_index()

    effectifs = df[cat_col].value_counts().reset_index()
    effectifs.columns = [cat_col, "count"]
    merged_data = effectifs.merge(tx_rsq, on=cat_col).sort_values(by=cible, ascending=True)
    fig, ax1 = plt.subplots(figsize=(4,4))
    ax2 = ax1.twinx()
    sns.barplot(data=merged_data, x=cat_col, y="count", color='grey', ax=ax1)
    sns.lineplot(data=merged_data, x=cat_col, y=cible, color='red', marker="o", ax=ax2)
    plt.show()


def cramer_V(cat_var1,cat_var2):
    crosstab = np.array(pd.crosstab(cat_var1,cat_var2,rownames=None,colnames=None)) #tableau de contingence
    stat = chi2_contingency(crosstab)[0] #stat de test de khi-2
    obs=np.sum(crosstab) 
    mini = min(crosstab.shape)-1 #min entre les colonnes et ligne du tableau croisé ==> ddl
    return (np.sqrt(stat/(obs*mini)))

def table_cramerV(df):
    rows=[]
    for var1 in df :
        col=[]
        for var2 in df :
            cramers = cramer_V(df[var1],df[var2])
            col.append(round(cramers,2))
        rows.append(col)
    cramers_results = np.array(rows)
    result=pd.DataFrame(cramers_results,columns=df.columns,index=df.columns)

def compute_cramers_v(df, categorical_vars, target):
    results = []
    for var1 in categorical_vars :  # Unpack index and column name
        if var1 == target:
            continue  # Skip the calculation if the variable is the target itself
        cv = cramer_V(df[var1], df[target])  # Correctly pass the variable names
        results.append([var1, cv])  # Append the variable name, not the tuple

    # Create a DataFrame to hold the results
    result_df = pd.DataFrame(results, columns=['Columns', "Cramer_V"])
    return result_df



from scipy.stats import chi2_contingency

def stats_liaisons_var_quali(df,categorical_columns):
    cramer_v_df = pd.DataFrame(index=categorical_columns,columns=categorical_columns)
    p_value_df = pd.DataFrame(index=categorical_columns,columns=categorical_columns)
    #tschuprow_t_df = pd.DataFrame(index=categorical_columns,columns=categorical_columns)

    #test de chi-deux pour chaque paire de variables quali
    for i, column1 in enumerate(categorical_columns):
        for j, column2 in enumerate(categorical_columns):
            if column1 != column2:
                contingency_table = pd.crosstab(df[column1], df[column2])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                cramer_v = np.sqrt(chi2 / (df.shape[0] * (min(contingency_table.shape)-1) ))
                #tschuprow_t = np.sqrt(chi2 / (df.shape[0] * np.sqrt((contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1))))
                cramer_v_df.loc[column1,column2] =cramer_v
                #tschuprow_t_df.loc[column1,column2] =tschuprow_t
                p_value_df.loc[column1,column2] = p
                
    return (p_value_df, cramer_v_df)


def filter_correlated_variables(df, threshold=0.7):
    # Replace NaNs with zeros if necessary (adjust based on your specific need)
    df.fillna(0, inplace=True)
    
    # Find indices where absolute correlation exceeds the threshold, ignoring the diagonal
    high_corr = (df.abs() > threshold).values & ~np.eye(df.shape[0], dtype=bool)
    
    # Extract pairs
    high_corr_pairs = [(df.index[i], df.columns[j]) for i, j in zip(*np.where(high_corr))]
    
    return high_corr_pairs

import itertools
# Fonction pour effectuer plusieurs cominaisons des éléments d'une liste
def combinaisons(liste_var,nb_elt):
    combinaisons = itertools.combinations(liste_var,nb_elt)
    liste_combinaisons = [c for c in list(combinaisons)]
    return liste_combinaisons

### Tests de outlier

def is_outlier(var):
    q1 = np.quantile(var, 0.25)
    q3 = np.quantile(var, 0.75)    
    IQR = q3 - q1
    
    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR

    # Retourne un tableau de booléens, chaque au moins un élément est True si c'est un outlier
    return any((var < lower_bound) | (var > upper_bound))


def list_columns_with_outliers(df, columns):
    outlier_columns = []
    
    for column in columns:
        if column in df.columns:
            # Convertir la colonne en tableau NumPy
            var = df[column].values
            
            # Vérifier s'il y a des outliers
            if is_outlier(var):
                outlier_columns.append(column)
    
    return outlier_columns

from scipy.stats.mstats import winsorize
def winsorization(df, columns, limits=(0.05, 0.05)):
    df_winsorized = df[columns].copy()  # Crée une copie du DataFrame pour éviter de modifier l'original
    
    for column in columns:
        if column in df_winsorized.columns:
            # Convertir la colonne en tableau NumPy
            var = df_winsorized[column].values
            
            # Vérifier s'il y a des outliers
            if is_outlier(var):
                # Appliquer la winsorisation
                df_winsorized[column] = winsorize(var, limits=limits)
    
    return df_winsorized








import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency

def discretize_variable(df, var_continuous, cible, n_classes=5):
    # 1. Découper en N classes
    df['class_bins'] = pd.qcut(df[var_continuous], q=n_classes, duplicates='drop')

    # 2. Calcul du taux de risque pour chaque classe
    risk_rates = df.groupby('class_bins')[cible].mean()

    # 3. Regrouper les classes selon les critères
    grouped_classes = []
    cumulative_weight = 0
    group = []

    for i, (interval, risk) in enumerate(risk_rates.items()):
        group.append(interval)
        cumulative_weight += df[df['class_bins'] == interval].shape[0] / df.shape[0]

        # Regrouper les classes pour que chaque groupe contienne au moins 5% de la population
        if cumulative_weight >= 0.05 or i == len(risk_rates) - 1:
            grouped_classes.append(group)
            group = []
            cumulative_weight = 0

    # 4. Retenir les transformations avec un écart de risque d'au moins 30%
    final_groups = []
    for group in grouped_classes:
        if len(group) > 1:
            group_risks = risk_rates[group]
            relative_risk_diff = np.abs(group_risks.max() - group_risks.min()) / group_risks.mean()

            if relative_risk_diff >= 0.3:
                final_groups.append(group)

    # 5. Vérifier les stabilités en volume et en risque (simulé ici, car dépend des données)
    stable_groups = []
    for group in final_groups:
        if len(group) > 1:
            stable_groups.append(group)

    # 6. Calculer les indicateurs statistiques T de Tschuprow et V de Cramer
    def tschuprow_t(x, y):
        contingency = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        return np.sqrt(chi2 / (n * min(contingency.shape)))

    def cramers_v(x, y):
        contingency = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        return np.sqrt(chi2 / (n * min(contingency.shape)))

    tschuprow_t_values = []
    cramers_v_values = []

    for group in stable_groups:
        df['grouped_bins'] = pd.cut(df[var_continuous], bins=np.array([g.left for g in group] + [group[-1].right]), include_lowest=True)
        tschuprow_t_values.append(tschuprow_t(df['grouped_bins'], df[cible]))
        cramers_v_values.append(cramers_v(df['grouped_bins'], df[cible]))

    return stable_groups, tschuprow_t_values, cramers_v_values


#def processing(X_test,categorical_variables,numerical_variables) :
# Continuous variables
    

# Application de modèles
import statsmodels.api as sm 
from sklearn.metrics import auc
from sklearn.metrics  import roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

def reg_logistique(df_train,var_cible,categorical_variables,numerical_variables):
    risk_drivers = categorical_variables + numerical_variables

    modalites_reference = []
    for var in categorical_variables :
        freq_defaut = df_train.groupby([var],as_index=True)[[var_cible]].mean().reset_index().sort_values(by=[var_cible],ascending=True).reset_index(drop=True)
        modalites_reference.append(var+'_'+str(freq_defaut[var][0]))

    # Transformer la base train
    #dummies_test = pd.get_dummies(df_test,columns=categorical_variables).drop(modalites_reference,axis=1).copy()
    dummies_train = pd.get_dummies(df_train,columns=categorical_variables,dtype='int').drop(modalites_reference,axis=1).copy()

    X_train = dummies_train.drop(var_cible,axis=1).copy()
    Y_train = dummies_train[var_cible].copy()

    #X_test = dummies_test.drop(var_cible,axis=1).copy()
    #Y_test = dummies_test[var_cible].copy()

    # Rajout de la constante dans le modèle
    X_train = sm.add_constant(X_train)
    #X_test = sm.add_constant(X_test)

    # Modèle
    model = sm.Logit(Y_train,X_train).fit(disp=False)
    y_train_pred = model.predict(X_train)
    #y_test_pred = model.predict(X_test)

    # Calculer les AUC
    fpr_train,tpr_train,thresholds_train = roc_curve(Y_train,y_train_pred)
    roc_auc_train = auc(fpr_train,tpr_train)

    #fpr_test,tpr_test,thresholds_test = roc_curve(Y_test,y_test_pred)
    #roc_auc_test = auc(fpr_test,tpr_test)

    # pvaleurs
    pvaleurs_coeffs = model.pvalues
    pvaleur_model = model.llr_pvalue

    flag_significativite = 0
    if (all(pvaleurs_coeffs<=0.05)) & (pvaleur_model<=0.05):
        flag_significativite = 1

    # VIF
    vif=pd.DataFrame()
    vif["variable"]=X_train.columns
    vif["VIF"]=[variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]
    vif.drop(0,inplace=True)

    # flag pour vérifier si tous les VIF sont <10
    flag_VIF = 0
    if all(vif["VIF"]<10):
        flag_VIF = 1

    return risk_drivers, pvaleur_model, pvaleurs_coeffs.to_dict(),flag_significativite,vif,flag_VIF,roc_auc_train, fpr_train,tpr_train,model


def selection_best_model(df_train,var_cible,categorical_variables,numerical_variables) :
    nb_var = len(categorical_variables+numerical_variables)

    #initialiser df pour stocker résultats
    resultat = pd.DataFrame(columns = ["variables","pvaleur_model","pvaleurs_coeffs","Significativité model et coeffs","VIF","flag_VIF","AUC_train"])

    #combinaisons possibles des risk drivers
    list_combinaisons = combinaisons(categorical_variables+numerical_variables,nb_var)
    for combinaison in list_combinaisons :
        model = reg_logistique(df_train[list(combinaison)+[var_cible]],var_cible,categorical_variables,numerical_variables)
        resultat = resultat.concat(
            {"variables" : model[0],
             "pvaleur_model" : model[1],
             "pvaleurs_coeffs" : model[2],
             "Significativité model et coeffs" : model[3],
             "VIF" : model[4],
             "flag_VIF" : model[5],
             "AUC_train" : model[6]},
            ignore_index = True)
            #"AUC_test" : model[7],


    resultat = resultat.sort_values(by=["Significativité model et coeffs","AUC_train","flag_VIF"],ascending=False).reset_index(drop=True)
    return resultat


#DF HAS NO ATTRIBUTES APPEND



    





























    
