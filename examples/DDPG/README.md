# DDPG (Deep Deterministic Policy Gradient)

## Explication

* Fusion entre le DQN (Deep Q-Learning) et une architecture actor-critic

Solutions :

1. **Experience Replay** : On stock l'expérience (S, A, R, S_next) dans une mémoire tampon de relecture (Replay Buffer) et on échantillone des minibatchs pour entaîner le réseau. Cela permet de décoreller les données et d'améliorer l'efficacité. Au début, le tampon de relecture est rempli d'expériences aléatoires.

2. **Target Network** : Ce réseau cible a la même architecture que l'approximateur de fonctions mais avec des paramètres figés. A chaque pas T (un hyperparamètre) les paramètres du réseau Q sont copiés sur le réseau cible. Cela conduit à un entraînement plus stable parce qu'il maintient la fonction cible fixe (pendant un certain temps).

3. **Actor-Critic** : "The Actor-Critic learning algorithm is used to represent the policy function independently of the value function. The policy function structure is known as the actor, and the value function structure is referred to as the critic. The actor produces an action given the current state of the environment, and the critic produces a TD (Temporal-Difference) error signal given the state and resultant reward. If the critic is estimating the action-value function Q(s,a), it will also need the output of the actor. The output of the critic drives learning in both the actor and the critic. In Deep Reinforcement Learning, neural networks can be used to represent the actor and critic structures." https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

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