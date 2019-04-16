# DDPG (Deep Deterministic Policy Gradient)

## Explication

* Fusion entre le DQN (Deep Q-Learning) et une architecture actor-critic
* L'utilisation d'un réseau neuronal non linéaire est puissante, mais l'entraînement est instable si on l'applique naïvement.

Solution :

1. **Experience Replay** : On stock l'expérience (S, A, R, S_next) dans une mémoire tampon de relecture (Replay Buffer) et on échantillone des minibatchs pour entaîner le réseau. Cela permet de décoreller les données et d'améliorer l'efficacité. Au début, le tampon de relecture est rempli d'expériences aléatoires.

2. **Target Network** : Ce réseau cible a la même architecture que l'approximateur de fonctions mais avec des paramètres figés. A chaque pas T (un hyperparamètre) les paramètres du réseau Q sont copiés sur le réseau cible. Cela conduit à un entraînement plus stable parce qu'il maintient la fonction cible fixe (pendant un certain temps).

## Utilisation

0. **Mettre le dossier "osim" à la racine du projet**
1. **activation de l'environnement** : `activate opensim-rl`
2. **phase d'entrainement** : `python DDPG.py --train`, options possibles : 
    - `--visualize` : permet de lancer le rendu de la simulation (False par défaut)
    - `--model X` : X à remplacer par le numéro du modèle (0 par défaut)
    - `--step Y` : Y à remplacer par le nombre de step sur lequel on veut s'entrainer (100000 par défaut)
3. **phase de test** : `python DDPG.py --test`, options possibles : 
    - `--visualize` : permet de lancer le rendu de la simulation (False par défaut)
    - `--model X` : X à remplacer par le numéro du modèle (0 par défaut)
    - `--step Y` : Y à remplacer par le nombre de step sur lequel on veut faire le test (100 par défaut, 3 avec la visualisation activée)