# SAC

## Utilisation

0. **activation de l'environnement** : `activate opensim-rl`
1. **prerequis**
'sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev'
'pip install stable-baselines'

2. **phase d'entrainement** : `python run.py --train`, options possibles : 
    - `--visualize` : permet de lancer le rendu de la simulation (False par défaut)
    - `--model X` : X à remplacer par le nom du modèle ("default" par défaut)
    - `--step Y` : Y à remplacer par le nombre de step sur lequel on veut s'entrainer (100000 par défaut)
3. **phase de test** : `python DDPG.py --test`, options possibles : 
    - `--visualize` : permet de lancer le rendu de la simulation (False par défaut)
    - `--model X` : X à remplacer par le nom du modèle (0 par défaut)
