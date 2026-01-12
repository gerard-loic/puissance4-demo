import random
from board import Board, PLAYER_ONE, PLAYER_TWO, EMPTY_CELL
import utils
from utils import ReplayBuffer
from simple_neural_network import SimpleNeuralNetwork
import time
import numpy as np

_drl_player_training_mode = None
_drl_player_network = None
_drl_player_target_network = None
_time_last_updated = None

class RandomPlayerCF:
    def __init__(self, position:int):
        self.position = position

    @staticmethod
    def play(board:Board):
        return random.choice(utils.get_available_moves_cf(board))
    


class HumanPlayerCF:
    def __init__(self, position:int):
        self.position = position

    @staticmethod
    def play(board:Board):
        while True:
            try:
                #Demande à l'utilisateur ou est ce qu'il veut jouer
                move = input("Enter the column that you would like to play (ex. A) :  ").strip().upper()
                col = ord(move[0]) - ord('A')

                if all(board.board[col][row] != EMPTY_CELL for row in range(board.rows)):
                    print("Column is full, try again")
                else:
                    for row in range(board.rows-1, -1, -1):
                        if board.board[col][row] == EMPTY_CELL:
                            return (row, col)
            except (IndexError, ValueError):
                print("This input is not correct !")
    

class HeuristicPlayer:
    def __init__(self, position:int):
        self.position = position

    def play(self, board:Board):
        best_move = None
        best_score = float('-inf')
        available_moves = utils.get_available_moves_cf(board=board)

        random.shuffle(available_moves) #Pour ne pas toujours sortir les memes coups au début
        for row, col in available_moves:
            board.play_move(row=row, col=col)   #On joue le coup
            score = utils.heuristic_evaluation(board=board, from_player_perspective=self.position)
            board.board[col][row] = EMPTY_CELL    #Annule le coup
            if score > best_score:
                best_score = score
                best_move = (row, col)

        return best_move


class LookAheadheuristicPlayer:
    def __init__(self, position:int, max_depth:int=2):
        self.position = position
        self.max_depth = max_depth
    
    def play(self, board:Board):
        #minimax algo to choose the best move
        best_move = None
        best_score = float('-inf')

        available_moves = utils.get_available_moves_cf(board=board)
        for row, col in available_moves:
            board.play_move(row=row, col=col)   #on joue le coup
            board.next_player() #Switch player (pour le prochain niveau)
            score = self._minimax(board=board, depth=0, is_maximizing=False)    #is_maximizing : Dépendant du niveau ou on est, cad du joueur qui joue
            board.board[col][row] = EMPTY_CELL  #Annule le coup
            board.next_player() #On retourne au joueur par défaut

            if score > best_score:
                best_score = score
                best_move = (row, col)
        
        return best_move
    
    def _minimax(self, board:Board, depth:int, is_maximizing:bool):
        #Vérifie si le jeu est fini
        game_ended, there_is_a_winner = board.game_has_ended()
        if game_ended:
            if there_is_a_winner:
                #On retourne le score, dépendant de si c'est le joueur ou l'adversaire
                return 1 if not is_maximizing else -1
            return 0    #Egalité
        
        if depth >= self.max_depth: #Si on a atteint le niveau max, on retourne le score de la situation
            return utils.heuristic_evaluation(board=board, from_player_perspective=self.position)
        
        #Coups dispos à partir de cette position
        available_moves = utils.get_available_moves_cf(board=board)
        best_score = float('-inf') if is_maximizing else float('inf')
        for row, col in available_moves:
            board.play_move(row=row, col=col)
            board.next_player()
            score = self._minimax(board=board, depth=depth+1, is_maximizing=not is_maximizing)  #Pour aller à N niveaux
            board.board[col][row] = EMPTY_CELL
            board.next_player()
            #Le meilleur score est négatif ou positif dépendant de la situation dans laquelle on est
            if is_maximizing:
                best_score = max(score, best_score)
            else:
                best_score = min(score, best_score)

        return best_score



class CustomLookAheadheuristicPlayer:
    def __init__(self, position:int, max_depth:int=2):
        self.position = position
        self.max_depth = max_depth
    
    def play(self, board:Board):
        #minimax algo to choose the best move
        best_move = None
        best_score = float('-inf')

        available_moves = utils.get_available_moves_cf(board=board)
        for row, col in available_moves:
            board.play_move(row=row, col=col)   #on joue le coup
            board.next_player() #Switch player (pour le prochain niveau)
            score = self._minimax(board=board, depth=0, is_maximizing=False)    #is_maximizing : Dépendant du niveau ou on est, cad du joueur qui joue
            board.board[col][row] = EMPTY_CELL  #Annule le coup
            board.next_player() #On retourne au joueur par défaut

            if score > best_score:
                best_score = score
                best_move = (row, col)
        
        return best_move
    
    def _minimax(self, board:Board, depth:int, is_maximizing:bool):
        #Vérifie si le jeu est fini
        game_ended, there_is_a_winner = board.game_has_ended()
        if game_ended:
            if there_is_a_winner:
                #On retourne le score, dépendant de si c'est le joueur ou l'adversaire
                return 1 if not is_maximizing else -1
            return 0    #Egalité
        
        if depth >= self.max_depth: #Si on a atteint le niveau max, on retourne le score de la situation
            return utils.heuristic_evaluation_custom(board=board, from_player_perspective=self.position)
        
        #Coups dispos à partir de cette position
        available_moves = utils.get_available_moves_cf(board=board)
        best_score = float('-inf') if is_maximizing else float('inf')
        for row, col in available_moves:
            board.play_move(row=row, col=col)
            board.next_player()
            score = self._minimax(board=board, depth=depth+1, is_maximizing=not is_maximizing)  #Pour aller à N niveaux
            board.board[col][row] = EMPTY_CELL
            board.next_player()
            #Le meilleur score est négatif ou positif dépendant de la situation dans laquelle on est
            if is_maximizing:
                best_score = max(score, best_score)
            else:
                best_score = min(score, best_score)

        return best_score

class DeepReenforcementLearningPlayer:
    def __init__(self, 
                position:int, 
                max_depth:int=2, 
                replay_buffer_max_size:int=5000, 
                replay_batch_size:int=64,
                alpha:float=0.1, 
                gamma:float=0.99, 
                epsilon:float=0.1, 
                training_mode:bool=False, 
                net_learnign_rate:float=0.005, 
                trained_network_file:str="trained_neuronal_network.pkl",
                update_target_every:int=8
                ):
        self.position = position
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_target_every = update_target_every
        self.trained_network_file = trained_network_file
        self.trained_target_network_file = trained_network_file.replace(".pkl", "_target.pkl")
        self.steps_until_last_update = 0
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_max_size, batch_size=replay_batch_size)

        global _drl_player_training_mode, _drl_player_network, _drl_player_target_network
        if _drl_player_target_network is None or _drl_player_training_mode is None or _drl_player_target_network is None:
            #Initialiser
            _drl_player_training_mode = training_mode
            
            #On définit le réseau de neurones
            sizes = [6*7, 148, 96, 1] #6 lignes et 7 colonnes puis 148 neurones, 96 neurones, et une valeur en couche de sortie
            activation_fct = ["relu", "tanh", "tanh"]
            iterations = 160

            _drl_player_network = SimpleNeuralNetwork(
                layers_sizes=sizes,
                activation_functions=activation_fct,
                alpha=net_learnign_rate,
                iterations=iterations,
                file_name=self.trained_network_file, 
                silent=True
            )

            _drl_player_target_network = SimpleNeuralNetwork(
                layers_sizes=sizes,
                activation_functions=activation_fct,
                alpha=net_learnign_rate,
                iterations=iterations,
                file_name=self.trained_target_network_file,
                silent=True
            )
        
        self.training_mode = training_mode
        self.network = _drl_player_network
        self.target_network = _drl_player_target_network


    def save_weights(self):
        global _time_last_updated
        now = time.time()

        if _time_last_updated is None or now-_time_last_updated >= 600:
            self.network.save()
            self.target_network.save()
            _time_last_updated = time.time()

    def _board_to_state(self, board:Board):
        #Convertir le board en un format compris par le réseau de neurones
        equiv_board = utils.equivalent_board_representation_cf(board=board)
        flatten_board = np.array(equiv_board.board).flatten() #On applatit en un tableau plat de 6*7 soit 42 éléments

        #np.where(condition, valeur_si_vrai, valeur_si_faux)
        return np.where(flatten_board == PLAYER_ONE, 1, np.where(flatten_board == PLAYER_TWO, -1, 0)) #Cette ligne transforme le plateau de jeu en valeurs numériques pour le réseau de neurones

    def play(self, board:Board):
        #Save the current state
        current_state = self._board_to_state(board=board)

        #Actions possibles
        available_moves = utils.get_available_moves_cf(board=board)

        #On les mélange
        random.shuffle(available_moves)

        #On regarde si on peut gagner en un seul mouvement (si c'est le cas pas besoin de réseau de neuroness)
        for row, col in available_moves:
            board.play_move(row=row, col=col)
            board.next_player()

            game_ended, there_is_a_winner = board.game_has_ended()
            if game_ended:
                #On récupère le nouvel état qu'on va vouloir sauvegarder
                next_state = self._board_to_state(board=board)

                #Undo the move
                board.board[col][row] = EMPTY_CELL
                board.next_player()

                if self.training_mode:
                    if there_is_a_winner:
                        #See it only from player ONE perspective
                        reward = 1 if self.position == PLAYER_ONE else -1
                    else:
                        reward = 0

                    self.replay_buffer.add_experience(
                        state=current_state,
                        action=(row, col),
                        reward=reward,
                        next_state=next_state,
                        done=True
                    )
                return row, col
            else:
                #Undo the move
                board.board[col][row] = EMPTY_CELL
                board.next_player()

        #Choisir un mouvement avec une stratégie epsilon-greedy
        if self.training_mode and np.random.rand() < self.epsilon:
            #On réalise une exploration (mouvement au hasard)
            best_move = random.choice(available_moves)
        else:
            #Exploitation
            best_score = float("-inf")
            best_move = None

            for row, col in available_moves:
                board.play_move(row=row, col=col)
                board.next_player()

                score = self._minimax(board=board, depth=0, is_maximazing=False)

                board.board[col][row] = EMPTY_CELL
                board.next_player()

                if score > best_score:
                    best_score = score
                    best_move = (row, col)

        #Sauvegarder dans le buffer
        if self.training_mode:
            row, col = best_move
            board.play_move(row=row, col=col)
            board.next_player()
            next_state = self._board_to_state(board=board)

            board.board[col][row] = EMPTY_CELL
            board.next_player()

            reward = 0
            done = False

            self.replay_buffer.add_experience(
                state=current_state,
                action=best_move,
                reward=reward,
                next_state=next_state,
                done=done
            )

        return best_move
    
    def _minimax(self, board:Board, depth:int, is_maximazing:bool):
        #On vérifie si le jeu est fini
        game_ended, there_is_a_winner = board.game_has_ended()

        if game_ended:
            if there_is_a_winner:
                return 1. if not is_maximazing else -1.
            return 0.
        
        if depth >= self.max_depth:
            network_input_state = self._board_to_state(board=board)
            score = float(self.network.forward(network_input_state))
            
            #On s'adapte au point de vue
            if self.position == PLAYER_ONE:
                return score
            else:
                return -1 * score

        #On avance dans l'arbre
        availaible_moves = utils.get_available_moves_cf(board=board)
        best_score = float("-inf") if is_maximazing else float("inf")
        for row, col in availaible_moves:
            board.play_move(row=row, col=col)
            board.next_player()

            score = self._minimax(board=board, depth=depth+1, is_maximazing=not is_maximazing)
            
            #On annule le coup
            board.board[col][row] = EMPTY_CELL
            board.next_player()

            #On garde le meilleure score en fonction qu'on cherche à maximiser ou minimiser
            if is_maximazing:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)

        return best_score
    
    def update(self):
        #On fait rien si on est pas en mode entrainement ou si le buffer n'est pas prêt
        if not self.training_mode or not self.replay_buffer.is_ready():
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch()
        state_values = self.network.forward(input_matrix=states).flatten()
        next_states_values = self.target_network.forward(input_matrix=next_states).flatten()
        next_states_targets = rewards + self.gamma * (1 - dones) * next_states_values

        #UPDATE TD formula
        # V(s) = V(s) + alpha * (reward + gamma * next_value - V(s))
        targets = state_values + self.alpha * (next_states_targets - state_values)

        #Train the network on the sample experience
        self.network.train(
            inputs=states,
            labels=targets,
            with_dropout=True, 
            silent=True
        )

        #On met à jour petit à petit le target network
        self.steps_until_last_update += 1
        if self.steps_until_last_update % self.update_target_every == 0:
            #On met à jour
            self._soft_update_target_network()
            self.steps_until_last_update = 0
    
    def _soft_update_target_network(self, tau=0.1):
        #Soft update target network weights

        for i in range(len(self.network.weights)):
            self.target_network.weights[i] = tau * self.network.weights[i] + (1 - tau) * self.target_network.weights[i]




