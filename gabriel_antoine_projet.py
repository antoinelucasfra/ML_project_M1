#!/usr/bin/env python
# coding: utf-8

# # Projet Machine Learning
# LUCAS Antoine     
# BESOMBES Gabriel     
# mars 2019     
# #    

# ## Introduction
# ---
# On s'intéresse a des données sur des maladies cardiaques. Le but étant de pouvoir prédire à partir d'un certain nombre de données si une personne a une maladie cardiaque ou non.   
# Nous allons étudier rapidement le jeu de données puis tester différents classifieurs issus de la library sklearn.
# 
# ---
# #    

# ## Importations
# ---
# Nous allons utiliser différentes library :
# * pandas pour ouvrir et visualiser nos données en format csv
# * matplotlib pour faire des graphiques
# * sklearn pour les différents classifieurs et leur vérification

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics


# ---
# #     

# ## Importation des données
# ---
# On ouvre le fichier _heart.csv_ qui se trouve à la racine de notre programme avec *pandas.read_csv()*   
# Ce fichier est disponible ici : https://www.kaggle.com/ronitf/heart-disease-uci

# In[2]:


heart = pd.read_csv("heart.csv")


# On visualise rapidement le début et la fin de notre dataframe

# In[3]:


heart.head()


# In[4]:


heart.tail()


# ---
# #     

# ## Test rapide sur le format
# ---
# On utilise _describe_ pour avoir un résumé rapide de nos données

# In[5]:


heart.describe()


# On vérifie qu'il ne manque pas de données :
# * _isnull()_ regarde si il manque une valeur
# * _any(axis=1)_ renvoie une liste avec True si il manque une valeur, False sinon. Les données sont traitées par ligne    
# 
# Si il manquait des valeurs on aurait des True en sortie

# In[6]:


for i in heart.isnull().any(axis=1):
    if i:
        print(i)


# ---
# #    

# ## Test avec les K plus proches voisins
# ---
# On teste rapidement avec un classifieur instancié à n=3

# In[7]:


knn = KNeighborsClassifier(n_neighbors=3)


# ###   
# On a laissé les autres paramètres à leurs valeurs par défaut comme on peut le voir ici :

# In[8]:


print(knn)


# ###    
# On regarde nos colonnes pour sortir celle qui correspond à la cible

# In[9]:


heart.columns


# ###    
# Ici il s'agit de la colonne target qui, comme spécifié sur kaggle, est composée de 1 et 0 suivant que la personne a eu une maladie cardiaque ou non.

# In[10]:


heart["target"].head()


# ###    
# Dans la suite on va utiliser *%%time* pour avoir une idée du temps d'exécution.

# ###   
# On sépare notre jeu de données en deux (70/30)

# In[11]:


get_ipython().run_cell_magic('time', '', 'X_train, X_test, y_train, y_test = train_test_split(heart.drop("target", axis=1),\n                                                    heart["target"],\n                                                    test_size=0.3)')


# ###   
# On entraine notre classifieur

# In[12]:


get_ipython().run_cell_magic('time', '', 'knn.fit(X_train, y_train)')


# ###    
# On fait une prédiction avec notre classifieur sur les données de test

# In[13]:


get_ipython().run_cell_magic('time', '', 'res=knn.predict(X_test)')


# ###    
# On calcule l'accurary de notre classifieur

# In[14]:


get_ipython().run_cell_magic('time', '', 'metrics.accuracy_score(y_test, res)')


# ###    
# On calcule la matrice de confusion

# In[15]:


cm = metrics.confusion_matrix(y_test, res)
print(cm)


# ###    
# On affiche avec pyplot

# In[16]:


fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=["Malades", "Sains"],
       yticklabels=["Malades", "Sains"],
       title="Matrice de confusion",
       ylabel='Etat réel',
       xlabel='Etat prédit')


plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")


thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()


# On constate que les erreurs se font surtout sur les sains (faux positifs):     
# Pour les sains on a:
# * 26 bonnes prédictions et 14 mauvaises, soit un taux d'erreur de 14/40\*100=35%
# 
# Pour les malades on a:
# * 47 bonnes prédictions et 4 mauvaises, soit un taux d'erreur de 4/51=\*100=8%
# 
# Ce type d'erreur est moins grave puisque ce serait vérifié par la suite par un médecin (Il vaut mieux en vérifier trop que pas assez)

# ###   
# On peut maintenant répéter l'opération plusieurs fois avec différentes valeurs de n pour voir quel n correspond le mieux

# In[17]:


l = []
knns = []
data = heart.drop("target", axis=1)
target = heart["target"]
for n in range(1,51):
    l.append([])
    knns.append([])
    for i in range(0,100):
        knn = KNeighborsClassifier(n_neighbors=n)
        X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=0.3)
        knn.fit(X_train, y_train)
        knns[-1].append([knn, y_test, knn.predict(X_test)])
        l[-1].append(metrics.accuracy_score(y_test, knn.predict(X_test)))


# In[18]:


l2 = [sum(X)/len(X) for X in l]


# In[19]:


plt.plot(l2)


# ###    
# On regarde pour quelle valeur de n on a la moyenne d'accuracy la plus haute.

# In[20]:


print(max(l2))
print(l2.index(max(l2)))


# Il s'agit donc de n=19

# ###   
# Puis on regarde quel classifieur a la meilleure accuracy

# In[21]:


print(max(max(l)))
print(max(l).index(max(max(l))))


# Ici c'est n=1

# In[22]:


print(l[0].index(max(l[0])))


# ###   
# On récupère ce classifieur puis on refait la matrice de confusion

# In[23]:


knn, y_test, res = knns[0][8]


# In[24]:


cm = metrics.confusion_matrix(y_test, res)
print(cm)


# ###    
# On affiche avec pyplot

# In[25]:


fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=["Malades", "Sains"],
       yticklabels=["Malades", "Sains"],
       title="Matrice de confusion",
       ylabel='Etat réel',
       xlabel='Etat prédit')


plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")


thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()


# On a le même type d'erreur mais en plus petites quantités ce qui correspond bien avec l'accuracy plus élevée.

# ---
# #    

# ## Réflexion intermédiaire
# ---
# La technique des voisins les plus proches n'est pas forcément la meilleure technique pour ce genre de jeux de données avec beaucoup de colonnes.    
# Nous allons donc tester d'autres classifieurs.
# 
# ---
# #     

# ## Test avec régression
# ---
# On teste rapidement avec une régression

# In[26]:


logreg = LogisticRegression(random_state=0,
                            solver='lbfgs',
                            multi_class='multinomial',
                            max_iter=5000)


# ###   
# On a laissé les paramètres à leurs valeurs par défaut comme on peut le voir ici :

# In[27]:


print(logreg)


# ###   
# On sépare notre jeu de données en deux (70/30)

# In[28]:


get_ipython().run_cell_magic('time', '', 'X_train, X_test, y_train, y_test = train_test_split(heart.drop("target", axis=1),\n                                                    heart["target"],\n                                                    test_size=0.3)')


# ###   
# On entraine notre classifieur

# In[29]:


get_ipython().run_cell_magic('time', '', 'logreg.fit(X_train, y_train)')


# ###    
# On fait une prédiction avec notre classifieur sur les données de test

# In[30]:


get_ipython().run_cell_magic('time', '', 'res=logreg.predict(X_test)')


# ###    
# On calcule l'accurary de notre classifieur

# In[31]:


get_ipython().run_cell_magic('time', '', 'metrics.accuracy_score(y_test, res)')


# ###   
# On peut maintenant répéter l'opération plusieurs fois pour voir si on peut avoir un meilleur modèle

# In[32]:


l = []
logregs = []
data = heart.drop("target", axis=1)
target = heart["target"]
for i in range(0,100):
    logreg = LogisticRegression(random_state=0,
                        solver='lbfgs',
                        multi_class='multinomial',
                        max_iter=5000)
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                target,
                                                test_size=0.3)
    logreg.fit(X_train, y_train)
    logregs.append([logreg, y_test, logreg.predict(X_test)])
    l.append(metrics.accuracy_score(y_test, logreg.predict(X_test)))


# In[33]:


plt.plot(l)


# In[34]:


sum(l)/len(l)


# On voit que sur les 100 modèles, certains ont une accuracy très haute et que la moyenne est de 82%. Ce classifieur a l'air de mieux marcher que le précédent

# ###   
# Puis on regarde quel classifieur a la meilleure accuracy

# In[35]:


print(max(l))
print(l.index(max(l)))


# ###   
# On récupère ce classifieur puis on refait la matrice de confusion

# In[36]:


logreg, y_test, res = logregs[84]


# In[37]:


cm = metrics.confusion_matrix(y_test, res)
print(cm)


# ###    
# On affiche avec pyplot

# In[38]:


fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=["Malades", "Sains"],
       yticklabels=["Malades", "Sains"],
       title="Matrice de confusion",
       ylabel='Etat réel',
       xlabel='Etat prédit')


plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")


thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()


# On a encore moins d'erreur et plus de faux positifs que de faux négatifs

# ---
# #    

# ## Naive Bayes
# ---

# In[39]:


GNB = GaussianNB()


# In[40]:


get_ipython().run_cell_magic('time', '', 'X_train, X_test, y_train, y_test = train_test_split(heart.drop("target", axis=1),\n                                                    heart["target"],\n                                                    test_size=0.3)')


# ###   
# On entraine notre classifieur

# In[41]:


get_ipython().run_cell_magic('time', '', 'GNB.fit(X_train, y_train)')


# ###    
# On fait une prédiction avec notre classifieur sur les données de test

# In[42]:


get_ipython().run_cell_magic('time', '', 'res=GNB.predict(X_test)')


# ###    
# On calcule l'accurary de notre classifieur

# In[43]:


get_ipython().run_cell_magic('time', '', 'metrics.accuracy_score(y_test, res)')


# L'accuracy semble plus faible mais le temps d'exécution est court

# ---
# #    

# ## Vector classification
# ---

# In[44]:


svc = SVC(kernel="linear")


# In[45]:


get_ipython().run_cell_magic('time', '', 'X_train, X_test, y_train, y_test = train_test_split(heart.drop("target", axis=1),\n                                                    heart["target"],\n                                                    test_size=0.3)')


# ###   
# On entraine notre classifieur

# In[46]:


get_ipython().run_cell_magic('time', '', 'svc.fit(X_train, y_train)')


# ###    
# On fait une prédiction avec notre classifieur sur les données de test

# In[47]:


get_ipython().run_cell_magic('time', '', 'res=svc.predict(X_test)')


# ###    
# On calcule l'accurary de notre classifieur

# In[48]:


get_ipython().run_cell_magic('time', '', 'metrics.accuracy_score(y_test, res)')


# On a de nouveau une accuracy élevée mais un temps d'exécution plus long

# ---
# #    

# ## Conclusion
# ---
# On constate que les modèles qui demandent le plus de ressources sont aussi plus précis dans notre cas. Mais aussi que différents classifieurs sont adaptés à différents types de jeux de données. Avec plus de données il faudrait donc trouver un compromis entre vitesse d'exécution et performance.   
# Dans notre cas des classifieurs plus simples et linéaires ont donnés de meilleurs résultats.
# 
# ---
# #    
