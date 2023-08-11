#implementation of a simple q-learning algorithm in a 3x4 gridworld
#loading dependencies
import numpy as np

#implementing grid world
class GridWorld:
    def __init__(self):
        '''
        Initialisierung einer einfachen 3x4 Gridworld
        die einzelnen Felder werden mit den Zahlen 0,1,-1 belegt
        0: normales Feld
        1: Ziel
        -1: Hindernis
        '''
        self.grid = np.array([
            [0,0,0,1],
            [0,0,0,-1],
            [0,0,0,0]
        ])
        self.num_rows, self.num_cols = self.grid.shape
        #Anzahl der möglichen Zustände:
        self.num_states = self.num_rows * self.num_cols
        self.state = (0,0) #initialer Zustand oben links

    def reset(self):
        '''
        setzt den Agenten auf den initialen Zustand zurück
        '''
        self.state = (0,0)
        return self._state_to_index(self.state) #gibt den initialen Zustand als Index zurück
    
    def step(self, action):
        '''
        führt eine Aktion aus und gibt den neuen Zustand, 
        die Belohnung und den Status des Spiels zurück
        '''
        if action==0: #nach rechts gehen
            self.state = (self.state[0], 
                          min(self.state[1]+1, self.num_cols-1))#nicht über den Rand hinausgehen
        elif action==1: #nach unten gehen
            self.state = (min(self.state[0]+1, self.num_rows-1),self.state[1])
        elif action==2: #nach links gehen
            self.state = (self.state[0], max(self.state[1]-1, 0))
        elif action==3: #nach oben gehen
            self.state = (max(self.state[0]-1, 0), self.state[1])

        #Belohnung berechnen
        row, col = self.state
        reward = self.grid[row, col]
        done = reward != 0 #Spiel ist beendet, wenn Belohnung ungleich 0 ist
        return self._state_to_index(self.state), reward, done
    
    def _state_to_index(self, state):
        '''
        wandelt einen Zustand in einen Index um
        dient der Vereinfachung der Berechnungen
        '''
        row, col = state
        return row * self.num_cols + col
    
    def display_grid(self,Q):
        solved_gird = np.zeros_like(self.grid, dtype=str)#leeres Grid

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if self.grid[row, col] !=0:
                    #Ziel oder Hindernis: Ende des Spiels
                    solved_gird[row, col] = 'T' if self.grid[row, col] > 0 else 'X'
                else:
                    #normales Feld: Aktion mit dem höchsten Q-Wert auswählen
                    state_idx = self._state_to_index((row, col))
                    action = np.argmax(Q[state_idx, :])
                    if action==0:
                        solved_gird[row, col] = '→'
                    elif action==1:
                        solved_gird[row, col] = '↓'
                    elif action==2:
                        solved_gird[row, col] = '←'
                    elif action==3:
                        solved_gird[row, col] = '↑'
        print("Solved Gridworld:")
        print(solved_gird)


#implementierung des Q-Learning Algorithmus
def q_learning(env, num_episodes, learning_rate, discount_factor, exploration_prob):
    '''
    env: die Gridworld-Umgebung (instanz von GridWorld)
    num_episodes: Anzahl der Episoden, die trainiert werden sollen
    learning_rate: Lernrate alpha
    discount_factor: Discount-Faktor gamma
    exploration_prob: Wahrscheinlichkeit, dass eine zufällige Aktion ausgeführt wird ansatt den gelernten Q-Wert zu nutzen (epsilon)
    '''
    num_states = env.num_states
    num_actions = 4 #Anzahl der möglichen Aktionen: rechts, unten, links, oben
    #initialisierung der Q-Tabelle
    Q = np.zeros((num_states, num_actions))

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            #Aktion auswählen: Epsilon-Greedy-Policy
            if np.random.random() < exploration_prob:
                action = np.random.randint(num_actions)#Explore: zufällige Aktion auswählen
            else:
                action = np.argmax(Q[state])#Exploit: Aktion mit dem höchsten Q-Wert auswählen

            #Aktion ausführen
            next_state, reward, done = env.step(action)

            #Q-Wert aktualisieren
            best_next_action = np.argmax(Q[next_state, :])
            #Q-Learning Update: Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
            Q[state, action] += learning_rate * (reward + discount_factor * Q[next_state, best_next_action] - Q[state, action])

            state = next_state #Zustand aktualisieren

    return Q

#Ausführen des Q-Learning Algorithmus
if __name__=="__main__":
    env = GridWorld()

    num_episodes = 1000
    learning_rate = 0.1
    discount_factor = 0.9
    exploration_prob = 0.1

    Q = q_learning(env, num_episodes, learning_rate, discount_factor, exploration_prob)
    print("Learned Q-values:\n",Q)
    print("")
    env.display_grid(Q)


