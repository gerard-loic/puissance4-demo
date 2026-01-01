from board import EMPTY_CELL, Board, PLAYER_ONE, PLAYER_TWO
import utils
import random
from copy import deepcopy
import os
import pickle

#Storage partagé entre les joueurs TemporalDifference
_state_values_for_td_players = {}

class RandomPlayer:
    def __init__(self, position:int):
        self.position = position

    @staticmethod
    def play(board:Board):
        return random.choice(utils.get_available_moves(board))
    
class HumanPlayer:
    def __init__(self, position:int):
        self.position = position

    @staticmethod
    def play(board:Board):
        while True:
            try:
                #Demande à l'utilisateur ou est ce qu'il veut jouer
                move = input("Enter your move. Ex A1:  ").strip().upper()
                col = ord(move[0]) - ord('A')
                row = int(move[1:]) - 1
                if board.board[col][row] == EMPTY_CELL:
                    return (row, col)
                else:
                    print("Cell is already occupied !")
            except (IndexError, ValueError):
                print("This input is not correct !")

#Implémente l'algo minimax
class MinimaxPlayer:
    def __init__(self, position:int):
        self.position = position
        self.stats = 0

    def play(self, board:Board):
        #Minimax algo pour choisir la meilleure action
        best_score = float('-inf')
        best_move = None

        #on itère sur les actions possibles
        available_moves = utils.get_available_moves(board)
        for row, col in available_moves:
            #On joue l'action, puis on change de joueur
            board.play_move(row=row, col=col)
            board.next_player()
            #On calcule le score de l'action avec l'algo minimax
            #Ici c'est à FALSE car on se place dans le point de vue de l'adversaire, qui cherche à MINIMISER le score
            score = self._minimax(board=board, is_maximizing=False)

            #Si le score est meilleur, on met à jour
            if score > best_score:
                best_score = score
                best_move = (row, col)

            #On annule l'action effectuée sur le plateau
            board.board[col][row] = EMPTY_CELL
            board.next_player()
        
        return best_move

    def _minimax(self, board:Board, is_maximizing:bool):
        self.stats += 1

        #Vérifie si le jeu est terminé
        game_ended, there_is_a_winner = board.game_has_ended()
        if game_ended:
            if there_is_a_winner:
                #Si player = winner ET is_maximizing=FALSE : on retourne 1
                return 1 if not is_maximizing else -1
            #Si égalité : retourne 0
            return 0
        
        #Si le jeu n'est pas terminé, on regarde les actions possibles
        available_moves = utils.get_available_moves(board)

        best_score = float('-inf') if is_maximizing else float('inf')
        for row, col in available_moves:
            #On joue l'action, puis on change de joueur
            board.play_move(row=row, col=col)
            board.next_player()
            #On calcule le score de l'action avec l'algo minimax
            score = self._minimax(board=board, is_maximizing= not is_maximizing)

            if is_maximizing:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)

            #On annule l'action effectuée sur le plateau
            board.board[col][row] = EMPTY_CELL
            board.next_player()

        return best_score


class DynamicprogrammingPlayer:
    def __init__(self, position:int):
        self.position = position
        #Mémoire des états
        self.state_values = {}
        self.stats = 0

    def play(self, board:Board):
        #Dynamic programming algo pour choisir la meilleure action
        best_score = float('-inf')
        best_move = None

        #on itère sur les actions possibles
        available_moves = utils.get_available_moves(board)
        for row, col in available_moves:
            #On joue l'action, puis on change de joueur
            board.play_move(row=row, col=col)
            board.next_player()
            #On calcule le score de l'action avec l'algo minimax
            #Ici c'est à FALSE car on se place dans le point de vue de l'adversaire, qui cherche à MINIMISER le score
            score = self._minimax_cached(board=board, is_maximizing=False)

            #Si le score est meilleur, on met à jour
            if score > best_score:
                best_score = score
                best_move = (row, col)

            #On annule l'action effectuée sur le plateau
            board.board[col][row] = EMPTY_CELL
            board.next_player()

        return best_move
    
    def _minimax_cached(self, board:Board, is_maximizing:bool):
        equiv_board = utils.equivalent_board_representation(board=board)
        state_key = repr(equiv_board)

        #L'etat n'est pas encore rencontré
        if state_key not in self.state_values:
            self.stats += 1
            #Vérifie si le jeu est terminé
            game_ended, there_is_a_winner = board.game_has_ended()
            if game_ended:
                if there_is_a_winner:
                    #Si player = winner ET is_maximizing=FALSE : on retourne 1
                    best_score = 1 if not is_maximizing else -1
                else:
                    #Si égalité : retourne 0
                    best_score = 0
            else:
                #Si le jeu n'est pas terminé, on regarde les actions possibles
                available_moves = utils.get_available_moves(board)

                best_score = float('-inf') if is_maximizing else float('inf')
                for row, col in available_moves:
                    #On joue l'action, puis on change de joueur
                    board.play_move(row=row, col=col)
                    board.next_player()
                    #On calcule le score de l'action avec l'algo minimax
                    score = self._minimax_cached(board=board, is_maximizing= not is_maximizing)

                    if is_maximizing:
                        best_score = max(best_score, score)
                    else:
                        best_score = min(best_score, score)

                    #On annule l'action effectuée sur le plateau
                    board.board[col][row] = EMPTY_CELL
                    board.next_player()
            self.state_values[state_key] = best_score

        return self.state_values[state_key]
        

class MontecarloPlayer:
    def __init__(self, position:int, num_simulations:int=5000):
        self.position = position
        self.num_simulations = num_simulations
        self.stats = 0

    def play(self, board:Board):
        #Monte Carlo algo pour choisir le meilleur coup à jouer

        best_move = None
        best_win_rate = float('-inf')

        available_moves = utils.get_available_moves(board=board)

        for row, col in available_moves:
            wins = 0
            #On joue le coup et on passe au joueur suivant
            board.play_move(row = row, col=col)
            board.next_player()

            game_ended, there_is_a_winner = board.game_has_ended()
            if game_ended:
                self.stats += 1
                if there_is_a_winner:
                    #Annulation
                    board.board[col][row] = EMPTY_CELL
                    board.next_player()
                    #Pas besoin d'aller plus loin, le coup est gagnant
                    return row, col
                if best_move is None:
                    best_move = (row, col)
            else:
                #On effectue les simulations pour estimer le win rate pour chaque état
                for _ in range(self.num_simulations):
                    wins += self._simulate_game(board=board)

                win_rate = wins / self.num_simulations
                #Si le taux est le meilleur, c'est peut etre le meilleur coup
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_move = (row, col)

            #Réinitialiser l'état
            board.board[col][row] = EMPTY_CELL
            board.next_player()

        return best_move
    
    def _simulate_game(self, board:Board):
        #On simule un jeu en commençant par un été spécifique (c'est à dire l'état du plateau)
        new_board = Board(
            rows=board.rows, 
            cols=board.cols, 
            connexions_to_win=board.connexions_to_win,
            board=deepcopy(board.board),
            player_to_move=board.current_player
        )

        opponent_move = False
        while True:
            #La boucle va simuler une partie complète. (le return termine la partie)
            available_moves = utils.get_available_moves(board=new_board)
            random_move = random.choice(available_moves)    #Choisit une action au hasard

            self.stats += 1

            #On joue le coup 
            new_board.play_move(row=random_move[0], col=random_move[1])
            
            game_ended, there_is_a_winner = new_board.game_has_ended()

            if game_ended:
                if there_is_a_winner:
                    return 1 if opponent_move else -1 #Retourne 1 si c'est au tour de l'adversaire et donc que le dernier coup était gagnant, -1 si c'est l'inverse
                return 0    # C'est une égalité
            
            #On passe au joueur suivant
            new_board.next_player()
            opponent_move = not opponent_move


class TemporaldifferencePlayer:
    def __init__(self, position:int, alpha:float=0.1, gamma:float=0.9, epsilon:float=0.1, training_mode:bool=False, state_values_filename:str="td_player_values.pkl"):
        self.position = position
        self.alpha = alpha  #Learning rate
        self.gamma = gamma  #Discount factor for futur rewards
        self.epsilon = epsilon #Initial exploration rate
        self.state_values_filename = state_values_filename  #Fichier où seront stockés les données d'entraînement
        self.training_mode = training_mode
        self.stats = 0

        #Chargement des états depuis le fichier si il existe
        global _state_values_for_td_players

        if os.path.exists(self.state_values_filename) and len(_state_values_for_td_players) == 0:
            with open(self.state_values_filename, 'rb') as f:
                _state_values_for_td_players = dict(pickle.load(f))

    @property
    def state_values(self):
        global _state_values_for_td_players
        return _state_values_for_td_players

    #Enregistre les nouveaux états
    def _save_new_state_values(self, state_key, state_value):
        global _state_values_for_td_players
        _state_values_for_td_players[state_key] = state_value
        with open(self.state_values_filename, 'wb') as f:
            pickle.dump(_state_values_for_td_players, f)


    def play(self, board:Board):
        available_moves = utils.get_available_moves(board=board)

        #On vérifie si on peut terminer le jeu en un seul coup (dans ce cas, aucun besoin d'explorer)
        for row, col in available_moves:
            board.play_move(row=row, col=col)

            game_ended, there_is_a_winner = board.game_has_ended()

            #Cancel move
            board.board[col][row] = EMPTY_CELL

            if game_ended:
                self.stats += 1
                #Mise à jour state value avec la formule TD:
                # V(s) = V(s) + alpha * (Reward + gamma * next_value - V(s))
                #For next value = 0 (car fin du jeu):
                # V(s) = V(s) + alpha * (Reward - V(s)) = (1 - alpha) * V(s) + alpha * Reward
                if there_is_a_winner:
                    reward = 1 if self.position == PLAYER_ONE else -1
                else:
                    reward = 0

                equiv_board = utils.equivalent_board_representation(board=board)
                state_key = repr(equiv_board)
                value = self.state_values.get(state_key, 0) #Valeur de départ à 0
                value = (1 - self.alpha) * value + self.alpha * reward  #Application de la formule
                self._save_new_state_values(state_key=state_key, state_value=value) #Sauvegarde

                return row, col
            
        #Choix de l'action basée sur epsilon-greedy stratégie
        if self.training_mode and random.random() < self.epsilon:
            #Exploration
            best_move = random.choice(available_moves)
            self.stats += 1
        else:
            #Exploitation
            #On va calculer que est le meilleur choix

            best_value = float('-inf')
            best_move = None
            for row, col in available_moves:
                self.stats += 1
                board.play_move(row=row, col=col) #On joue le coup
                board.next_player() #Switch joueurs

                #On veut calculer la valeur du coup
                equiv_board = utils.equivalent_board_representation(board=board)
                value = self.state_values.get(repr(equiv_board), 0)  #Si on a déjà enregistré une valeur on la récupère, sinon c'est 0
                value = (1 if self.position == PLAYER_ONE else 0) * value   #On adapte en négatif si player TWO

                #On annule le coup
                board.board[col][row] = EMPTY_CELL
                board.next_player()

                if value > best_value:
                    best_value = value
                    best_move = (row, col)
        
        #Après avoir choisi un coup, mise à jour de la valeur du coup
        equiv_board = utils.equivalent_board_representation(board=board)
        state_key = repr(equiv_board)
        value = self.state_values.get(state_key, 0)

        #On joue le coup
        board.play_move(row=best_move[0], col=best_move[1])
        board.next_player()

        #Etat après le coup
        equiv_board = utils.equivalent_board_representation(board=board)
        next_state_key = repr(equiv_board)

        next_value = self.state_values.get(next_state_key, 0)

        #On annule le coup
        board.board[best_move[1]][best_move[0]] = EMPTY_CELL
        board.next_player()

        #Mise à jour avec la formule
        # V(s) = V(s) + alpha * (Reward + gamma * next_value - V(s))
        #Ici reward = 0 : car dans un jeu de tictactoe on ne recçoit une récompense qu'à la fin, or on a déjà testé ce cas
        value = value + self.alpha * (self.gamma * next_value - value)
        self._save_new_state_values(state_key=state_key, state_value=value)

        return best_move



            

class TDPlayer(object):

    def __init__(self, *, position, alpha=0.1, gamma=0.9, epsilon=0.1, training_mode=False,
                 state_values_file_name='td_player_values.pkl'):
        self.position = position
        self.alpha = alpha  # Learning rate for Temporal Difference
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        self.state_values_file = state_values_file_name
        self.training_mode = training_mode
        self.stats = 0

        # Load state values from file, if exists, otherwise initialise as an empty dict
        global _state_values_for_td_players
        if os.path.exists(self.state_values_file) and len(_state_values_for_td_players) == 0:
            with open(self.state_values_file, 'rb') as f:
                _state_values_for_td_players = dict(pickle.load(f))

    @property
    def state_values(self):
        global _state_values_for_td_players
        return _state_values_for_td_players

    def save_new_state_value(self, state_key, new_value):
        global _state_values_for_td_players
        _state_values_for_td_players[state_key] = new_value

        with open(self.state_values_file, 'wb') as f:
            pickle.dump(_state_values_for_td_players, f)

    def play(self, *, board):
        available_moves = utils.get_available_moves(board)

        # See if the game can be terminated in one move
        for row, col in available_moves:
            board.play_move(row=row, col=col)
            game_ended, there_is_winner = board.game_has_ended()
            board.board[col][row] = EMPTY_CELL  # Undo move
            if game_ended:
                self.stats += 1
                # Update the state value with the TD formula:
                # V(s) = V(s) + alpha * ( Reward + gamma * next_value - V(s) )
                # For next_value = 0:
                # V(s) = V(s) + alpha * (Reward - V(s)) = (1-alpha) * V(s) + alpha * Reward
                if there_is_winner:
                    reward = 1 if self.position == PLAYER_ONE else -1
                else:
                    reward = 0
                equiv_board = utils.equivalent_board_representation(board)
                state_key = repr(equiv_board)
                value = self.state_values.get(state_key, 0)
                value = (1 - self.alpha) * value + self.alpha * reward
                self.save_new_state_value(state_key, value)
                return row, col

        # Choose move based on epsilon-greedy strategy
        if self.training_mode and random.random() < self.epsilon:
            # Exploration
            best_move = random.choice(available_moves)
            self.stats += 1
        else:
            # Exploitation
            best_value = float('-inf')
            best_move = None
            for row, col in available_moves:
                self.stats += 1
                board.play_move(row=row, col=col)
                board.next_player()  # Swit ch turn for players
                equiv_board = utils.equivalent_board_representation(board)
                value = (1 if self.position == PLAYER_ONE else -1) * self.state_values.get(repr(equiv_board), 0)
                board.board[col][row] = EMPTY_CELL  # Undo move
                board.next_player()  # Switch turn to original player
                if value > best_value:
                    best_value = value
                    best_move = (row, col)

        # After choosing a move, update the state values using TD formula
        equiv_board = utils.equivalent_board_representation(board)
        state_key = repr(equiv_board)
        value = self.state_values.get(state_key, 0)
        board.play_move(row=best_move[0], col=best_move[1])
        board.next_player()
        equiv_board = utils.equivalent_board_representation(board)
        next_state_key = repr(equiv_board)
        next_value = self.state_values.get(next_state_key, 0)
        board.board[best_move[1]][best_move[0]] = EMPTY_CELL  # Undo move
        board.next_player()

        # Update TD formula
        # V(s) = V(s) + alpha * ( Reward + gamma * next_value - V(s) )
        value = value + self.alpha * (self.gamma * next_value - value)
        self.save_new_state_value(state_key, value)
        return best_move
