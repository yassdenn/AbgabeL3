# LERNZIEL 3: Reinforcement Learning
## Lösung der CarRacing-v2 Umgebung mit Hilfe von Proximal Policy Optimization (PPO)

## 1. Einleitung:
In diesem Projekt werden zwei Modelle zur Lösung der CarRacing-v2-Umgebung trainiert. Beide Modelle basieren auf den PPO-Algorithmus, jedoch wird beim ersten das zugrundeliegende CNN durch die von Stable Baselines3 bereitgestellte CnnPolicy ersetzt. Das zweite Modell verwendet eine eigene Implementation basierend auf der CNN-Architektur von [Fakhry, 2020](https://towardsdatascience.com/applying-a-deep-q-network-for-openais-car-racing-game-a642daf58fc9). Die Modelle werden auf der CarRacing-v2-Umgebung innerhalb von 100 Episoden evaluiert. 

+Zusatzleistung: Implementierung des q-learning Algorithmus und Lösung einer einfachen 3x4 Grid-Umgebung.

## 2. Beschreibung der Datein:
- `project.ipynb`: Notebook zur Erkundung der CarRacing-v2-Umgebung + Evaluation des standard PPO-Modells
- `train_standard_model.ipynb`: Notebook zum Training des standard PPO-Modells
- `train_custom_model.py`: Notebook zum Training des custom PPO-Modells + Evaluation des custom PPO-Modells
- `q-learning.py`: Zusatzleistung
- `models`: Enthält die trainierten Modelle
- `logs`: Enthält die Log-Dateien der trainierten Modelle

## 3. standard PPO-Modell:
### Training
Das Training wird mit Hilfe von stable_baselines3 durchgeführt. Die Installation erfolgt über den folgenden Befehl:
```bash
pip install stable_baselines3
```
Festgelegte Trainingparameter:
- `policy`: CnnPolicy - Ausgewählt auf Grund der Tatsache, dass die Eingabe ein Bild ist. Dieses zeigt den aktuellen Verlauf der Rennstrecke.
- `env`: CartPole-v2 - Beschreibt das Environment, in dem das Modell trainiert wird. Es besteht aus einem Rennwagen und einer Rennstrecke. Das Ziel ist es, den Wagen so zu steuern, dass er nicht von der Strecke abkommt, wodurch man einen hohen Score erreicht.
- `total_timesteps`: 500.000 - Anzahl der Episoden, die das Modell trainiert wird.
### Evaluation
Die Evaluation erfolgt ebenfalls mit Hilfe von stable_baselines3. Die dafür eingesetzte Funktion lautet `evaluate_policy()`. Diese gibt den durchschnittlichen Reward des Modells zurück. Dieser wird über 100 Episoden berechnet. Die Evaluation erfolgt auf dem CartPole-v2 Environment. Das erreichte Ergebnis zeichnet sich wie folgt aus:
- `Mean reward`: 40.89
- `Standard deviation`: 80.52

## 4. custom PPO-Modell:
### Training
Auch hier wird die Implementation des PPO-Algorithmus von stable_baselines3 verwendet. Das zugrundeliegende CNN wird jedoch durch eine eigene Implementation ersetzt. Dies geschieht mit Hilfe von PyTorch. Die Installation erfolgt über den folgenden Befehl:
```bash
pip install torch
```
Festgelegte Trainingparameter:
- `policy`: CnnPolicy - Wobei die Architektur des Actors personalisiert wurde (siehe [training_custom_model.ipynb](training_custom_model.ipynb) class CustomCnn) 
- `env`: CarRacing-v2
- `total_timesteps`: 170.000 Episoden
### Evaluation
Auch hier wird mit `evaluate_policy()` der Durchschnittsreward und die Standardabweichung des Modells berechnet. Die Evaluation erfolgt auf dem CarRacing-v2 Environment. Das erreichte Ergebnis zeichnet sich wie folgt aus:
- `Mean reward`: 348.32
- `Standard deviation`: 181.38

## 5. Vergleich und Fazit:
Das custom PPO-Modell erreicht einen erheblich höheren Durchschnittsreward als das standard PPO-Modell. Dies ist auf die personalisierte Architektur des Actors zurückzuführen. Die Standardabweichung ist bei beiden Modellen sehr hoch. Dies ist auf die Tatsache zurückzuführen, dass die CarRacing-v2-Umgebung sehr komplex ist. 

## 6. Zusatzleistung:
[q-learning.py](q-learning.py): Implemetierung des q-learning Algorithumus und einer einfache 3x4 Grid-Umgebung. und Lösung dieser Umgebung mit Hilfe des q-learning Algorithmus.
