import math
import itertools
import numpy as np

from sklearn import metrics
from sklearn.model_selection import cross_val_score

class Selection_Grille:
    """
    GridSearchCV customisée. 
    """

    def __init__(self, model, param_grid, cv):
        """
        Attributs à initialiser: 
        - model :         Le modèle dont on souhaite optimiser les paramètres
        - param_grid :    Pour chaque paramètre du modèle, les diverses valeurs à envisager
        - cv :            Le nombre de folds de la validation croisée utilisée par la fonction. 
        - best_params_ :  Dictionnaire contenant, pour chaque paramètre, la valeur optimale
                          qui aura été choisie à l'issue de l'entraînement (fit).
        - cv_results_ :   Dictionnaire contenant, pour chaque combinaison possible de paramètres:
                          - La moyenne des scores obtenus lors de chaque étape de cross-validation 
                          - L'écart-type des scores obtenus lors de chaque étape de cross-validation
        - current_index : Paramètre technique utilisé par la fonction get_fold.
        """
        self.model_ = model
        self.param_grid_ = param_grid
        self.cv_ = cv

        self.best_params_ = dict()
        self.cv_results_ = dict()     
        self.current_index = 0

    def fit(self, X, y):
        """
        On choisit par validation croisée les paramètres qui permettent
        d'optimiser le score du modèle donné sur l'ensemble d'entraînement 
        constitué de X (variables) et y (cible).
        """
        n_samples = X.shape[0]  
        best_score = 0 

        self.cv_results_['mean_test_score'] = list()
        self.cv_results_['std_test_score'] = list()
        self.cv_results_['params'] = list()	       

        # On souhaite essayer toutes les combinaisons possibles des paramètres. 
        # Pour cela, on utilise le module itertools de python

        keys = self.param_grid_.keys()
        combinaisons = list()
        for k in keys:
            combinaisons.append(self.param_grid_[k])

        combinaisons = list(itertools.product(*combinaisons))

        # On parcourt toutes les combinaisons possibles
        for combinaison in combinaisons:

            # On assigne la combinaison considérée au modèle à optimiser       
            params = dict()
            i = 0
            for k in keys:
                params[k] = combinaison[i]
                i += 1

            self.model_.set_params(**params)

            # On évalue cette combinaison de paramètres via une validation croisée
            all_scores_cv = list()
            self.current_index = 0

            for cv in range(self.cv_):

                # Détermination du test set de ce fold
                start, end = self.get_fold(cv, n_samples)

                X_test_fold = X[start:end, :]
                y_test_fold = y[start:end]
                X_train_fold = np.concatenate((X[:start, :], X[end:, :]), axis = 0)
                y_train_fold = np.concatenate((y[:start], y[end:]), axis = 0)  

                self.model_.fit(X_train_fold, y_train_fold)
                y_pred_fold = self.model_.predict(X_test_fold)
                score_fold = metrics.accuracy_score(y_test_fold, y_pred_fold)
                all_scores_cv.append(score_fold)

            self.cv_results_['mean_test_score'].append(np.mean(all_scores_cv))
            self.cv_results_['std_test_score'].append(np.std(all_scores_cv))
            self.cv_results_['params'].append(params)

            # On sauvegarde la combinaison la plus performante jusqu'à présent
            if np.mean(all_scores_cv) > best_score:
                best_score = np.mean(all_scores_cv)
                i = 0
                for k in keys:
                    self.best_params_[k] = combinaison[i]
                    i += 1

        # On stocke le modèle le plus performant dans l'attribut self.model_
        self.model_.set_params(**self.best_params_)
        self.model_.fit(X, y)             

    def get_fold(self, cv, n_samples):
        """
        Fonction permettant de diviser le training set pour une étape de validation croisée. 
        Plus précisément: ramène l'index de départ et de fin du test set pour ce fold.

        Pour le choix du test-set pour ce fold, on reproduit le fonctionnement de la fonction 
        Kfolds du module model_selection de scikit-learn.             

        Dans l'aide, on peut lire: "The first n_samples % n_splits folds have size 
        n_samples // n_splits + 1, other folds have size n_samples // n_splits, 
        where n_samples is the number of samples."
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)

        C'est donc ce qu'on choisit d'implémenter ici. 
        """

        if cv < n_samples % self.cv_:
            size_fold = n_samples // self.cv_ + 1
        else:
            size_fold = n_samples // self.cv_

        # Découpage des données
        start = self.current_index
        end = self.current_index + size_fold

        self.current_index += size_fold

        return start, end

    def predict(self, X_test):
        """
        On effectue une prédiction sur le modèle préalablement optimisé avec la méthode fit.
        """
        return self.model_.predict(X_test)