import random
from board import Board, PLAYER_ONE, PLAYER_TWO, EMPTY_CELL
import utils
from utils import ReplayBuffer
from simple_neural_network import SimpleNeuralNetwork
from ts_simple_neural_network import TsSimpleNeuralNetwork
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



class DRLPlayer(object):
    def __init__(self, position, max_depth=2, replay_buffer_max_size=5000, replay_batch_size=64,
                 alpha_td=0.1, gamma=0.99, epsilon=0.1, training_mode=False,
                 net_learning_rate=0.005, trained_network_file='../trained_networks/trained_drl_network.pkl', update_every=8,
                 ):
        self.position = position
        self.max_depth = max_depth
        self.alpha = alpha_td
        self.gamma = gamma
        self.epsilon = epsilon
        self.steps_until_last_update = 0
        self.update_every = update_every
        self.trained_network_file = trained_network_file
        self.trained_target_network_file = trained_network_file.replace('.pkl', '_target.pkl')
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_max_size, batch_size=replay_batch_size)

        self.count = 0

        global _drl_player_network, _drl_player_target_network, _drl_player_training_mode
        if _drl_player_network is None or _drl_player_target_network is None or _drl_player_training_mode is None:
            _drl_player_training_mode = training_mode
            sizes = [6*7, 148, 96, 1]
            act_fns = ['relu', 'tanh', 'tanh']
            iterations = 160
            _drl_player_network = SimpleNeuralNetwork(layers_sizes=sizes, activation_functions=act_fns,
                                                      alpha=net_learning_rate, iterations=iterations, silent=True,
                                                      file_name=self.trained_network_file)
            _drl_player_target_network = SimpleNeuralNetwork(layers_sizes=sizes, activation_functions=act_fns,
                                                             alpha=net_learning_rate, iterations=iterations, silent=True,
                                                             file_name=self.trained_target_network_file)
        self.training_mode = _drl_player_training_mode
        self.network = _drl_player_network
        self.target_network = _drl_player_target_network

    def save_weights(self):
        print("SAVE")
        global _time_last_updated
        now = time.time()
        if _time_last_updated is None or now - _time_last_updated >= 600:
            self.network.save(self.trained_network_file)
            self.target_network.save(self.trained_target_network_file)
            _time_last_updated = time.time()

    @staticmethod
    def _board_to_state(board):
        # Convert the board to the neural network input
        equiv_board = utils.equivalent_board_representative_cf(board)
        flatten_board = np.array(equiv_board.board).flatten()
        return np.where(flatten_board == PLAYER_ONE, 1, np.where(flatten_board == PLAYER_TWO, -1, 0))

    def play(self, *, board):
        current_state = self._board_to_state(board)
        available_moves = utils.get_available_moves_cf(board)
        random.shuffle(available_moves)

        # See if the game can be won / draw in the next move
        for row, col in available_moves:
            board.play_move(row=row, col=col)
            board.next_player()
            game_ended, there_is_winner = board.game_has_ended()
            if game_ended:
                next_state = self._board_to_state(board)
                board.board[col][row] = EMPTY_CELL  # Undo move
                board.next_player()  # Switch to original player
                if self.training_mode:
                    if there_is_winner:
                        # See it only from the Player 1's perspective
                        reward = 1 if self.position == PLAYER_ONE else -1
                    else:
                        reward = 0
                    self.replay_buffer.add_experience(current_state, (row, col), reward, next_state, True)
                return row, col
            else:
                board.board[col][row] = EMPTY_CELL  # Undo move
                board.next_player()  # Switch to original player

        # Choose a move with simple epsilon-greedy strategy
        if self.training_mode and np.random.rand() < self.epsilon:
            # Exploration
            best_move = random.choice(available_moves)
        else:
            # Exploitation
            best_score = float('-inf')
            best_move = None
            for row, col in available_moves:
                board.play_move(row=row, col=col)
                board.next_player()
                score = self._minimax(board, depth=0, is_maximizing=False)
                board.board[col][row] = EMPTY_CELL  # Undo move
                board.next_player()  # Switch to original player
                if score > best_score:
                    best_score = score
                    best_move = (row, col)

        # Save experience to replay buffer
        if self.training_mode:
            row, col = best_move
            board.play_move(row=row, col=col)
            board.next_player()
            next_state = self._board_to_state(board)
            board.board[col][row] = EMPTY_CELL  # Undo move
            board.next_player()  # Switch to original player

            reward, done = 0, False
            self.replay_buffer.add_experience(current_state, best_move, reward, next_state, done)

        return best_move

    def _minimax(self, board, depth, is_maximizing):
        game_ended, there_is_winner = board.game_has_ended()
        if game_ended:
            if there_is_winner:
                return 1. if not is_maximizing else -1.
            return 0.

        if depth >= self.max_depth:
            network_input_state = self._board_to_state(board)
            score = float(self.network.forward(network_input_state))
            self.count += 1
            if self.position == PLAYER_ONE:
                return score
            else:
                return -1. * score

        available_moves = utils.get_available_moves_cf(board)
        best_score = float('-inf') if is_maximizing else float('inf')
        for row, col in available_moves:
            board.play_move(row=row, col=col)
            board.next_player()
            score = self._minimax(board, depth=depth+1, is_maximizing=not is_maximizing)
            board.board[col][row] = EMPTY_CELL  # Undo move
            board.next_player()  # Switch to original player
            if is_maximizing:
                best_score = max(score, best_score)
            else:
                best_score = min(score, best_score)

        return best_score

    def update(self):
        if not self.training_mode or not self.replay_buffer.is_ready():
            return

        states, _, rewards, next_states, dones = self.replay_buffer.sample_batch()
        state_values = self.network.forward(states).flatten()
        next_states_values = self.target_network.forward(next_states).flatten()
        next_state_targets = rewards + self.gamma * (1-dones) * next_states_values

        # Update TD formula: V(s) = V(s) + alpha * (Reward + gamma * next_value - V(s))
        targets = state_values + self.alpha * (next_state_targets - state_values)

        # Train the primary network on the sample experiences
        self.network.train(inputs=states, labels=targets, with_dropout=True, silent=True)

        # Slowly update the target network as well
        self.steps_until_last_update += 1
        if self.steps_until_last_update % self.update_every == 0:
            self._soft_update_target_network()
            self.steps_until_last_update = 0

    def _soft_update_target_network(self, tau=0.1):
        # Soft update target network weights
        for i in range(len(self.network.weights)):
            self.target_network.weights[i] = tau * self.network.weights[i] + (1 - tau) * self.target_network.weights[i]


class TsDRLPlayer(object):
    @staticmethod
    def enableExperienceCache():
        TsDRLPlayer.enabled = True
        TsDRLPlayer.cache = {}

    @staticmethod
    def clearExperienceCache():
        if hasattr(TsDRLPlayer, 'enabled'):
            TsDRLPlayer.cache = {}

    @staticmethod
    def addExperienceCache(key:str, value:tuple):
        if hasattr(TsDRLPlayer, 'enabled'):
            TsDRLPlayer.cache[key] = value

    @staticmethod
    def getExperienceCache(key:str):
        if hasattr(TsDRLPlayer, 'enabled') and key in TsDRLPlayer.cache:
            return TsDRLPlayer.cache[key]
        return None

    def __init__(self, position, max_depth=2, replay_buffer_max_size=5000, replay_batch_size=64,
                 alpha_td=0.1, gamma=0.99, epsilon=0.1, training_mode=False,
                 net_learning_rate=0.005, trained_network_file='../trained_networks/trained_ts_drl_network.keras', update_every=8,
                 ):
        self.position = position
        self.max_depth = max_depth
        self.alpha = alpha_td
        self.gamma = gamma
        self.epsilon = epsilon
        self.steps_until_last_update = 0
        self.update_every = update_every
        self.trained_network_file = trained_network_file
        self.trained_target_network_file = trained_network_file.replace('.keras', '_target.keras')
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_max_size, batch_size=replay_batch_size)

        self.count = 0

        global _drl_player_network, _drl_player_target_network, _drl_player_training_mode
        if _drl_player_network is None or _drl_player_target_network is None or _drl_player_training_mode is None:
            _drl_player_training_mode = training_mode

            _drl_player_network = TsSimpleNeuralNetwork(
                input_shape=(6*7,),
                layer_sizes=[148, 96, 1],
                activation_functions=['relu', 'tanh', 'tanh'],
                with_dropout=True, 
                loss_fct='mse', 
                metrics=['mae']
            )
            _drl_player_target_network = TsSimpleNeuralNetwork(
                input_shape=(6*7,),
                layer_sizes=[148, 96, 1],
                activation_functions=['relu', 'tanh', 'tanh'],
                with_dropout=True, 
                loss_fct='mse', 
                metrics=['mae']
            )

        self.training_mode = _drl_player_training_mode
        self.network = _drl_player_network
        self.target_network = _drl_player_target_network

    def save_weights(self):
        print("SAVE")
        global _time_last_updated
        now = time.time()
        if _time_last_updated is None or now - _time_last_updated >= 600:
            self.network.save(self.trained_network_file)
            self.target_network.save(self.trained_target_network_file)
            _time_last_updated = time.time()

    @staticmethod
    def _board_to_state(board):
        # Convert the board to the neural network input
        equiv_board = utils.equivalent_board_representative_cf(board)
        flatten_board = np.array(equiv_board.board).flatten()
        return np.where(flatten_board == PLAYER_ONE, 1, np.where(flatten_board == PLAYER_TWO, -1, 0))

    def play(self, *, board):
        current_state = self._board_to_state(board)
        available_moves = utils.get_available_moves_cf(board)
        random.shuffle(available_moves)

        # See if the game can be won / draw in the next move
        for row, col in available_moves:
            board.play_move(row=row, col=col)
            board.next_player()
            game_ended, there_is_winner = board.game_has_ended()
            if game_ended:
                next_state = self._board_to_state(board)
                board.board[col][row] = EMPTY_CELL  # Undo move
                board.next_player()  # Switch to original player
                if self.training_mode:
                    if there_is_winner:
                        # See it only from the Player 1's perspective
                        reward = 1 if self.position == PLAYER_ONE else -1
                    else:
                        reward = 0
                    self.replay_buffer.add_experience(current_state, (row, col), reward, next_state, True)
                return row, col
            else:
                board.board[col][row] = EMPTY_CELL  # Undo move
                board.next_player()  # Switch to original player

        # Choose a move with simple epsilon-greedy strategy
        if self.training_mode and np.random.rand() < self.epsilon:
            # Exploration
            best_move = random.choice(available_moves)
        else:
            # Exploitation
            best_score = float('-inf')
            best_move = None
            for row, col in available_moves:
                board.play_move(row=row, col=col)
                board.next_player()
                score = self._minimax(board, depth=0, is_maximizing=False)
                board.board[col][row] = EMPTY_CELL  # Undo move
                board.next_player()  # Switch to original player
                if score > best_score:
                    best_score = score
                    best_move = (row, col)

        # Save experience to replay buffer
        if self.training_mode:
            row, col = best_move
            board.play_move(row=row, col=col)
            board.next_player()
            next_state = self._board_to_state(board)
            board.board[col][row] = EMPTY_CELL  # Undo move
            board.next_player()  # Switch to original player

            reward, done = 0, False
            self.replay_buffer.add_experience(current_state, best_move, reward, next_state, done)

        return best_move
    """
    def _minimax(self, board, depth, is_maximizing):
        game_ended, there_is_winner = board.game_has_ended()
        if game_ended:
            if there_is_winner:
                return 1. if not is_maximizing else -1.
            return 0.

        if depth >= self.max_depth:
            network_input_state = self._board_to_state(board).reshape(1, -1)
            score = float(self.network.forward(network_input_state)[0, 0])
            self.count += 1
            if self.position == PLAYER_ONE:
                return score
            else:
                return -1. * score

        available_moves = utils.get_available_moves_cf(board)
        best_score = float('-inf') if is_maximizing else float('inf')
        for row, col in available_moves:
            board.play_move(row=row, col=col)
            board.next_player()
            score = self._minimax(board, depth=depth+1, is_maximizing=not is_maximizing)
            board.board[col][row] = EMPTY_CELL  # Undo move
            board.next_player()  # Switch to original player
            if is_maximizing:
                best_score = max(score, best_score)
            else:
                best_score = min(score, best_score)

        return best_score
    """
    def _minimax(self, board, depth, is_maximizing):
        game_ended, there_is_winner = board.game_has_ended()
        if game_ended:
            if there_is_winner:
                return 1. if not is_maximizing else -1.
            return 0.

        if depth >= self.max_depth:
            network_input_state = self._board_to_state(board).reshape(1, -1)
            score = float(self.network.forward(network_input_state)[0, 0])
            self.count += 1
            if self.position == PLAYER_ONE:
                return score
            else:
                return -1. * score

        available_moves = utils.get_available_moves_cf(board)
        
        # 🚀 OPTIMISATION: Batch evaluation quand on est juste avant max_depth
        if depth == self.max_depth - 1:
            positions_to_eval = []
            positions_from_cache = []
            move_to_score = {}  # {(row, col): score}
            
            for row, col in available_moves:
                board.play_move(row=row, col=col)
                board.next_player()
                
                # Vérifier si le jeu est terminé
                game_ended, there_is_winner = board.game_has_ended()
                
                if game_ended:
                    # Position terminale - calculer directement le score
                    if there_is_winner:
                        score = 1. if not is_maximizing else -1.
                    else:
                        score = 0.
                    move_to_score[(row, col)] = score
                else:
                    # Position non-terminale - ajouter pour batch evaluation
                    network_input_state = self._board_to_state(board)
                    cacheKey = repr(board)
                    cacheValue = TsDRLPlayer.getExperienceCache(key=cacheKey)
                    if cacheValue is None:
                        #Pas encore en cache, à évaluer
                        positions_to_eval.append((row, col, network_input_state, cacheKey))
                    else:
                        #Déjà en cache, on récupère la valeur
                        positions_from_cache.append(cacheValue)
                    
                
                board.board[col][row] = EMPTY_CELL  # Undo move
                board.next_player()
            
            # 🚀 Évaluer toutes les positions non-terminales en batch
            if positions_to_eval:
                batch_states = np.array([state for _, _, state, _ in positions_to_eval])
                batch_scores = self.network.forward_batch(batch_states).flatten()
                self.count += len(batch_scores)
                
                # Ajuster les scores selon la position du joueur
                if self.position == PLAYER_TWO:
                    batch_scores = -1. * batch_scores
                
                # Assigner les scores aux mouvements
                for i, (row, col, state, key) in enumerate(positions_to_eval):
                    move_to_score[(row, col)] = float(batch_scores[i])
                    raw_score = float(batch_scores[i]) if self.position == PLAYER_ONE else -float(batch_scores[i])

                    TsDRLPlayer.addExperienceCache(key=key, value=(row, col, state, raw_score))

            #Ajouter les scores provenant du cache
            for i, (row, col, _, score) in enumerate(positions_from_cache):
                if self.position == PLAYER_TWO:
                    score *= -1
                move_to_score[(row, col)] = float(score)
            
            # Trouver le meilleur score
            if is_maximizing:
                return max(move_to_score.values())
            else:
                return min(move_to_score.values())
        
        # 📝 Code normal pour les autres profondeurs (récursion classique)
        best_score = float('-inf') if is_maximizing else float('inf')
        for row, col in available_moves:
            board.play_move(row=row, col=col)
            board.next_player()
            score = self._minimax(board, depth=depth+1, is_maximizing=not is_maximizing)
            board.board[col][row] = EMPTY_CELL  # Undo move
            board.next_player()  # Switch to original player
            if is_maximizing:
                best_score = max(score, best_score)
            else:
                best_score = min(score, best_score)

        return best_score

    def update(self):
        if not self.training_mode or not self.replay_buffer.is_ready():
            return

        states, _, rewards, next_states, dones = self.replay_buffer.sample_batch()

        # Convertir en numpy arrays si nécessaire
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)


        # Utiliser le réseau principal (avec Dropout) pour les valeurs
        state_values = self.network.network(states, training=False).numpy().flatten()
        next_states_values = self.target_network.network(next_states, training=False).numpy().flatten()
        next_state_targets = rewards + self.gamma * (1-dones) * next_states_values

        targets = state_values + self.alpha * (next_state_targets - state_values)
        targets = targets.reshape(-1, 1)

        # Train the primary network on the sample experiences
        self.network.train(
            inputs=states,
            labels=targets,
            iterations=1,
            verbose=False
        )

        # Slowly update the target network as well
        self.steps_until_last_update += 1
        if self.steps_until_last_update % self.update_every == 0:
            self._soft_update_target_network()
            self.steps_until_last_update = 0


    def _soft_update_target_network(self, tau=0.1):
        # Soft update target network weights
        self.target_network.soft_update_from(self.network, tau=tau)



