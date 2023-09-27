import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""
# Prédiction de survie d'un passager du Titanic
## Auriez-vous survécu au Titanic?

Aspirant Data Scientist j'ai mis en place cette web application pour prédire si oui ou non vous auriez survécu au naufrage du Titanic.
"""

tab1, tab2, tab3 = st.tabs(["Exploration des données", "Le modèle", "A vous de jouer"])

with tab1:
	"""
	Le jeu de données dont je me suis servi pour mettre au point ce modèle est le fameux titanic.csv disponible en libre accès sur Kaggle.
	Si vous le souhaitez vous pouvez explorer les données dans le tableau juste en-dessous.
	"""

	df = pd.read_csv("titanic.csv")
	st.dataframe(df, use_container_width = True)

	#Distribution des âges
	st.subheader("Visualisation des données")
	fig, ax = plt.subplots(2, 2, figsize = (12, 10))
	ax[0, 0].hist(df['Age'].dropna(), bins=20, color='skyblue', edgecolor='black')
	ax[0, 0].set_xlabel("Âge")
	ax[0, 0].set_ylabel("Nombre de passagers")
	ax[0, 0].set_title("Distribution des âges")

	#Distribution des classes
	sns.countplot(data=df, x='Pclass', ax = ax[0, 1])
	ax[0, 1].set_title("Distribution des classes")
	ax[0, 1].set_xlabel("Classe")
	ax[0, 1].set_ylabel("Nombre de passagers")
	ax[0, 1].bar_label(ax[0, 1].containers[0])
	ax[0, 1].set_ylim(0, 550)

	#Distribution des survivants/non-survivants
	sns.countplot(data=df, x='Survived', ax=ax[1, 0])
	ax[1, 0].set_title("Survivants vs. Non-survivants")
	ax[1, 0].set_xlabel("")
	ax[1, 0].set_ylabel("Nombre de passagers")
	ax[1, 0].bar_label(ax[1, 0].containers[0])
	ax[1, 0].set_xticklabels(['Survivants', 'Non-survivants'])
	ax[1, 0].set_ylim(0, 600)

	#Répartition des passagers par sexe
	sns.countplot(data=df, x='Sex', ax=ax[1, 1])
	ax[1, 1].set_title("Répartition des passagers par sexe")
	ax[1, 1].set_xlabel("Sexe")
	ax[1, 1].set_ylabel("Nombre de passagers")
	ax[1, 1].bar_label(ax[1, 1].containers[0])
	ax[1, 1].set_xticklabels(['Mâle', 'Femelle'])
	ax[1, 1].set_ylim(0, 700)

	st.pyplot(fig)

	st.subheader("Corrélations entre les variables")
	df.dropna(subset=['Age', 'Embarked'], inplace=True)
	df.drop(['Cabin', 'Name', 'Ticket', 'Embarked'], inplace=True, axis = 1)
	df['Sex'] = df['Sex'].replace({'female': 0, 'male': 1})

	st.image("heatmap_titanic.png")

	"""
	Cette heatmap donne un aperçu de la relation entre 'Survived' et les autres caractéristiques
	du dataset.\n
	**Survived vs. Sex** : Il y a une corrélation négative significative (-0.54) entre la survie et le sexe. Cela signifie que les femmes (Sex = 0) ont une plus grande probabilité de survie que les hommes (Sex = 1).\n
	**Survived vs. Pclass** : Il y a une corrélation négative significative (-0.36) entre la survie et la classe des passagers (Pclass). Les passagers dans des classes de statut social plus élevé (Pclass = 1) ont une plus grande probabilité de survie.\n
	**Survived vs. Age** : La corrélation entre la survie et l'âge est légèrement négative (-0.08), ce qui indique une faible relation entre l'âge et la survie.\n
	**Survived vs. Fare**: Il y a une corrélation positive significative (0.27) entre la survie et le tarif payé (Fare). Cela suggère que les passagers ayant payé des tarifs plus élevés ont une plus grande probabilité de survie.\n
	**Survived vs. SibSp** et **Survived vs. Parch** : Les corrélations avec ces deux caractéristiques (nombre de frères et sœurs / conjoints à bord, nombre de parents / enfants à bord) sont relativement faibles.\n
	"""

with tab2:
	"""
	## Étape 1 : Préparation et division des données  

	Dans cette étape, j'ai préparé mes données pour l'entraînement du modèle. Voici ce que j'ai fait :

	J'ai sélectionné les caractéristiques pertinentes pour mon modèle, notamment 'Sex', 'Pclass', 'Age' et 'Fare'.
	J'ai transformé la colonne 'Sex' en valeurs numériques pour que le modèle puisse les comprendre.
	Pour évaluer mon modèle, j'ai divisé mes données en ensembles d'entraînement et de test. Cela m'a permis de mesurer la performance du modèle sur un ensemble de données indépendant. J'ai utilisé la fonction train_test_split pour réaliser cette division.

	"""
	prep = '''df['Sex'] = df['Sex'].replace({'female': 0, 'male': 1})
	data = df.drop(['Survived'], axis=1)
	target = df['Survived']

	data['Sex'] = data['Sex'].replace({'female': 0, 'male': 1})
	data = data[["Sex", "Pclass", "Age", "Fare"]]
	'''

	st.code(prep, language = 'python')

	"""
	## Étape 2 : Entraînement du modèle initial

	J'ai comparé les performances d'un modèle Support Vector Classification, d'un Random Forest Classifier (RFC) et d'un Gradient Boost Classifier.
	J'ai retenu modèle RFC. J'ai entraîné le modèle avec les hyperparamètres par défaut et j'ai évalué sa précision sur l'ensemble de test. La précision initiale était de 78%.
	"""
	rfc_1 = '''#Instanciation du modèle
	random_forest = RandomForestClassifier(n_estimators=100, random_state= 42)

	#Entrainement du modèle
	random_forest.fit(X_train, y_train)

	#Evaluation du modèle
	rfc_pred = random_forest.predict(X_test)
	accuracy_rfc = accuracy_score(y_test, rfc_pred)
	print("Précision :", accuracy_rfc)
	'''

	st.code(rfc_1, language='python')
	"""
	## Étape 3 : Optimisation des hyperparamètres

	Pour améliorer la performance du modèle, j'ai entrepris une recherche d'hyperparamètres à l'aide de GridSearchCV.
	"""

	grid_search = '''#Recherche des meilleurs hypereparamètres
	param_grid_rfc = {
   	'n_estimators': [50, 100, 200],              # Nombre d'arbres dans la forêt
   	'max_depth': [None, 10, 20, 30],            # Profondeur maximale des arbres
   	'min_samples_split': [2, 5, 10],           # Nombre minimum d'échantillons requis pour diviser un nœud
   	'min_samples_leaf': [1, 2, 4],             # Nombre minimum d'échantillons requis dans une feuille
   	'bootstrap': [True, False],                 # Si les échantillons sont bootstrapés ou non
   	'criterion': ['gini', 'entropy']           # Critère de fraction : 'gini' ou 'entropy'
	}

	grid_search_rfc = GridSearchCV(estimator=random_forest, param_grid=param_grid_rfc, cv=5, scoring='accuracy', n_jobs=-1)
	grid_search_rfc.fit(X_train, y_train)

	best_params_rfc = grid_search_rfc.best_params_
	best_score_rfc = grid_search_rfc.best_score_
	'''

	st.code(grid_search, language='python')
	
	"""
	Cette recherche a identifié les meilleures combinaisons d'hyperparamètres pour mon modèle RandomForest. Voici les résultats :

	Meilleurs hyperparamètres: {'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}

	J'ai donc ré-évalué le RFC avec ces paramètres et obtenu un score de prédiction de 81% (Le meilleur score pendant l'entrainement était de 83%).
	"""

	rfc_2 = '''# On récupère le meilleur estimateur (modèle) trouvé par la recherche d'hyperparamètres
	best_rfc = grid_search_rfc.best_estimator_

	# On utilise le meilleur modèle pour faire des prédictions sur l'ensemble de test
	y_pred_best_rfc = best_rfc.predict(X_test)

	# On calcule l'accuracy (précision) en comparant les prédictions aux vraies étiquettes (y_test)
	accuracy_best_rfc = accuracy_score(y_test, y_pred_best_rfc)

	'''

	st.code(rfc_2, language='python')
	"""
	## Étape 4 : Validation croisée

	J'ai finalement effectué une validation croisée en utilisant StratifiedKFold pour obtenir une évaluation plus robuste de la performance du modèle. 
	"""

	kfold = '''from sklearn.model_selection import StratifiedKFold

	# Créez un objet StratifiedKFold
	n_splits = 5  # Par exemple, 5 plis
	stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

	# Bouclez à travers les plis
	for train_index, test_index in stratified_kfold.split(data, target):
   	X_train, X_test = data.iloc[train_index], data.iloc[test_index]  # Utilisez .iloc pour accéder aux données par les indices
   	y_train, y_test = target.iloc[train_index], target.iloc[test_index]

	pred_rfc = best_rfc.predict(X_test)
	accuracy_best_rfc_2 = accuracy_score(y_test,pred_rfc)
	'''

	st.code(kfold, language = 'python')

	"""
	Après validation croisée le score du modèle atteint **95%**! Une amélioration significative.
	"""

with tab3:
   st.subheader("Simulation de survie au Titanic")

   """
   A vous de jouer! Ajustez les paramètres ci-dessous pour savoir si vous auriez
   survécu au naufrage du Titanic!
	"""


   #Ajouter des widgets pour entrer les données de simulation
   age = st.slider("**Âge**", min_value = 0, max_value = 100, value = 30, step = 1)

   """
   Utilisez ce curseur pour sélection votre âge en faisant glisser le surseur entre 0 et 100 ans.
   Vous pouvez également saisir l'âge directement dans la zone de texte. Le curseur a une valeur par défaut
   de 30 ans, mais vous pouvez le régler sur n'importe quelle valeur entre la plage spécifiée.
   """

   sex = st.selectbox("**Sexe**", ("Mâle", "Femelle"))
   """
	Sélectionnez votre sexe entre "Mâle" et "Femelle".
   """
   pclass = st.selectbox("**Classe**", [1, 2, 3])
   """
	Utilisez la liste déroulante pour choisir la classe à laquelle vous auriez appartenu parmi les
	options : 1 (première classe), 2 (seconde classe) ou 3 (troisième classe).
   """
   fare = st.number_input("**Tarif**", min_value = 0.0, max_value = 1000.0, value = 50.0, step = 10.0)
   """
   Utilisez la zone de saisie numérique pour entrer le tarif que vous auriez payé pour vorte billet.
   Vous pouvez régler le tarif en augmentant ou en diminuant la valeur à l'aide des flèches ou en saisissant
   une valeur directement. La valeur par défaut est de 50.0, mais vous pouvez la modifier en fonction
   de votre choix
   """
   #Gestion de l'évènement de bouton pour simuler la survie
   if st.button("Simuler la survie"):
   	#Préparer les données de simulation pour le modèle
   	sex_mapping = {"Mâle" : 1, "Femelle" : 0}
   	sim_data = pd.DataFrame({'Sex': [sex_mapping[sex]], 'Pclass': [pclass], 'Age': [age], 'Fare': [fare]})

   	#Charger le modèle RFC depuis le fichier
   	with open("meilleur_modele_rfc.pkl", "rb") as model_file:
   		loaded_rfc_model = joblib.load(model_file)

   	#Prédiction de la survie
   	survival_prediction = loaded_rfc_model.predict(sim_data)

   	#Affichage du résultat
   	if survival_prediction[0] == 1:
   		st.success("Vous auriez survécu au naufrage du Titanic!")
   	else:
   		st.error("Vous n'auriez pas survécu au naufrage du Titanic!")