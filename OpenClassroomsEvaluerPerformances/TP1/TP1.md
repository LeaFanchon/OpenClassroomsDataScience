### Introduction

L'objet de ce TP est de ré-implémenter la fonction de recherche sur grille de la librairie scikit-learn (la fonction model_selection.GridSearchCV), dans l’objectif d’effectuer la classification du dataset sur la qualité du vin.<br>
Les deux premières parties de ce document correspondent au TP du cours. C'est seulement à la troisième qu'on ilustrera le fonctionnement de la fonction de recherche sur grille ré-implémentée.
- **Première partie** (TP du cours) : Analyse et traitement préliminaire des données.
- **Deuxième partie** (TP du cours) : Sélection d'un modèle via la fonction GridSearchCV de Scikit-learn. 
- **Troisième partie** : Sélection d'un modèle via la fonction de recherche sur grille ré-implémentée, qu'on appellera la fonction *maison*. 

NB : Le code de la fonction ré-implémentée se trouve dans le fichier **selection_grille_maison.py** joint à ce devoir.

### I. Exploration et traitement préliminaire des données - TP du cours


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('winequality-white.csv', sep=";")
```


```python
X = data.as_matrix(data.columns[:-1])
y = data.as_matrix([data.columns[-1]])
y = y.flatten()
```

    /home/lea/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.
    /home/lea/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      



```python
y_class = np.where(y < 6, 0, 1)
```


```python
from sklearn import model_selection

# 30% des données dans le jeu de test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_class,test_size=0.3)
```


```python
from sklearn import preprocessing
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)
```

### II. Choix des paramètres via model_selection.GridSearchCV - TP du cours


```python
from sklearn import neighbors, metrics

# Fixer les valeurs des hyperparamètres à tester
param_grid = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15]}

# Choisir un score à optimiser, ici l'accuracy (proportion de prédictions correctes)
score = 'accuracy'

# Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
clf_skl = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), # un classifieur kNN
                 param_grid,                                             # hyperparamètres à tester
                 cv=model_selection.KFold(n_splits=5,shuffle=False),     # nb de folds de validation croisée
                 scoring=score                                           # score à optimiser
             )


# Optimiser ce classifieur sur le jeu d'entraînement
clf_skl.fit(X_train_std, y_train)

# Afficher le(s) hyperparamètre(s) optimaux
print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement pour model_selection.gridSearchCV:")
print(clf_skl.best_params_)

# Afficher les performances correspondantes
print("Résultats de la recherche sur grille en utilisant model_selection.gridSearchCV:")
for mean, std, params in zip(clf_skl.cv_results_['mean_test_score'], # score moyen
                             clf_skl.cv_results_['std_test_score'],  # écart-type du score
                             clf_skl.cv_results_['params']           # valeur de l'hyperparamètre
                            ):
    print("\t%s = %0.3f (+/-%0.03f) for %r" 
        % (score,    # critère utilisé
            mean,    # score moyen
            std * 2, # barre d'erreur
            params   # hyperparamètre
        ))
```

    Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement pour model_selection.gridSearchCV:
    {'n_neighbors': 3}
    Résultats de la recherche sur grille en utilisant model_selection.gridSearchCV:
    	accuracy = 0.767 (+/-0.023) for {'n_neighbors': 3}
    	accuracy = 0.760 (+/-0.012) for {'n_neighbors': 5}
    	accuracy = 0.759 (+/-0.022) for {'n_neighbors': 7}
    	accuracy = 0.759 (+/-0.030) for {'n_neighbors': 9}
    	accuracy = 0.757 (+/-0.043) for {'n_neighbors': 11}
    	accuracy = 0.756 (+/-0.041) for {'n_neighbors': 13}
    	accuracy = 0.758 (+/-0.027) for {'n_neighbors': 15}



```python
y_pred = clf_skl.predict(X_test_std)
print("\nSur le jeu de test, avec la fonction de scikit-learn : %0.3f" % metrics.accuracy_score(y_test, y_pred))
```

    
    Sur le jeu de test, avec la fonction de scikit-learn : 0.756


### III. Choix des paramètres via une fonction de recherche sur grille "maison"


```python
import selection_grille_maison as sg_maison
```


```python
from sklearn import neighbors, metrics

# Fixer les valeurs des hyperparamètres à tester
param_grid = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15]}

score = 'accuracy'

# Appel de la nouvelle fonction de sélection sur grille
clf_maison = sg_maison.Selection_Grille(neighbors.KNeighborsClassifier(), # un classifieur kNN
                                        param_grid, # hyperparamètres à tester
                                        cv=5 # nombre de folds de validation croisée
                                    )

clf_maison.fit(X_train_std, y_train)

# Afficher le(s) hyperparamètre(s) optimaux
print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement pour la fonction 'maison':")
print(clf_maison.best_params_)

# Afficher les performances correspondantes
print("Résultats de la recherche sur grille en utilisant la fonction 'maison' :")
for mean, std, params in zip(clf_maison.cv_results_['mean_test_score'], # score moyen
                             clf_maison.cv_results_['std_test_score'], # écart-type du score
                             clf_maison.cv_results_['params'] # valeur de l'hyperparamètre
                            ):
    print("\t%s = %0.3f (+/-%0.03f) for %r" 
        % (score, # critère utilisé
            mean, # score moyen
            std * 2, # barre d'erreur
            params # hyperparamètre
        ))
```

    Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement pour la fonction 'maison':
    {'n_neighbors': 3}
    Résultats de la recherche sur grille en utilisant la fonction 'maison' :
    	accuracy = 0.767 (+/-0.023) for {'n_neighbors': 3}
    	accuracy = 0.760 (+/-0.012) for {'n_neighbors': 5}
    	accuracy = 0.759 (+/-0.022) for {'n_neighbors': 7}
    	accuracy = 0.759 (+/-0.030) for {'n_neighbors': 9}
    	accuracy = 0.757 (+/-0.043) for {'n_neighbors': 11}
    	accuracy = 0.756 (+/-0.041) for {'n_neighbors': 13}
    	accuracy = 0.758 (+/-0.027) for {'n_neighbors': 15}



```python
y_pred = clf_maison.predict(X_test_std)
print("\nSur le jeu de test, avec la fonction 'maison' : %0.3f" % metrics.accuracy_score(y_test, y_pred))
```

    
    Sur le jeu de test, avec la fonction 'maison' : 0.756


### Conclusion
On voit que les résultats sont strictement identiques entre la fonction "maison" et la fonction model_selection.GridSearchCV. On a donc bien réussi à la ré-implémenter correctement. 
