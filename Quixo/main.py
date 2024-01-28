import random
from enum import Enum

from game import Game, Move, Player
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


class SharedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(SharedMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Hardswish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Hardswish(),
            nn.Dropout(p=0.5),
            # nn.Linear(hidden_size, hidden_size),
            # nn.Hardswish(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.Hardswish(),
            # nn.Dropout(p=0.5),
            nn.Linear(hidden_size, out_size),
            nn.Hardswish()
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.mlp(x)


class MyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, action_1_out_size, action_2_out_size):
        super(MyMLP, self).__init__()
        self.shared_mlp = SharedMLP(input_size, 128, hidden_size)
        self.model_action_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Hardswish(),
            nn.Linear(hidden_size, action_1_out_size)
        )
        self.model_action_2 = nn.Sequential(
            nn.Linear(hidden_size + action_1_out_size, hidden_size),
            nn.Hardswish(),
            nn.Linear(hidden_size, action_2_out_size)
        )
        for layer in self.model_action_1:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        for layer in self.model_action_2:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward_action1(self, x):
        shared_mlp_output = self.shared_mlp(x)
        return self.model_action_1(shared_mlp_output)

    def forward_action2(self, x):
        shared_mlp_output = self.shared_mlp(x)
        action_1_output = self.model_action_1(shared_mlp_output)
        combined_input = torch.cat((shared_mlp_output, action_1_output), dim=1)
        return self.model_action_2(combined_input)


class TrainingGame_dqn(Game):
    def __init__(self, board_state: np.array = None) -> None:
        super().__init__()
        if board_state is not None: self._board = deepcopy(board_state)

    def play(self, player1: Player, player2: Player) -> int:
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        current_player_idx = 1
        winner = -1
        while winner < 0:
            # get board old state
            old_state = self.get_board()

            current_player_idx += 1
            current_player_idx %= len(players)
            ok = False
            while not ok:
              from_pos, slide = players[current_player_idx].make_move(self)
              ok = self.move(from_pos, slide, current_player_idx)
              if not ok and isinstance(players[current_player_idx], MyPlayer_dqn):
                  # my model contains also invalid moves so I wanna learn not to do them
                  reward = players[current_player_idx].compute_reward(self.get_current_player(), old_state, old_state,-1, True)
                  players[current_player_idx].get_experience(old_state, players[current_player_idx].get_pos(from_pos),
                                                             slide, reward, self.get_board(), False, current_player_idx)
            winner = self.check_winner()

        if isinstance(players[current_player_idx], MyPlayer_dqn):
            # get board new state
            new_state = self.get_board()
            # 1 here the player should get the reward
            reward = players[current_player_idx].compute_reward(self.get_current_player(), old_state, new_state, winner == current_player_idx)
            # 2 here I would be capable of gettin my new step record, to put it into my replay memory
            players[current_player_idx].get_experience(old_state, players[current_player_idx].get_pos(from_pos),
                                                       slide, reward, new_state, winner == -1, current_player_idx)
        return winner

    def move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id > 2:
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.__take((from_pos[1], from_pos[0]), player_id)
        if acceptable:
            acceptable = self.__slide((from_pos[1], from_pos[0]), slide)
            if not acceptable:
                self._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable

    def __take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        '''Take piece'''
        # acceptable only if in border
        acceptable: bool = (
            # check if it is in the first row
            (from_pos[0] == 0 and from_pos[1] < 5)
            # check if it is in the last row
            or (from_pos[0] == 4 and from_pos[1] < 5)
            # check if it is in the first column
            or (from_pos[1] == 0 and from_pos[0] < 5)
            # check if it is in the last column
            or (from_pos[1] == 4 and from_pos[0] < 5)
            # and check if the piece can be moved by the current player
        ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
        if acceptable:
            self._board[from_pos] = player_id
        return acceptable

    def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        '''Slide the other pieces'''
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == 4 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == 4 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == (0, 0) and (
                slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == (4, 0) and (
                slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == (0, 4) and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        # if it is
        if acceptable:
            # take the piece
            piece = self._board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                self._board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], self._board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    self._board[(i, from_pos[1])] = self._board[(
                        i - 1, from_pos[1])]
                # move the piece up
                self._board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], self._board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    self._board[(i, from_pos[1])] = self._board[(
                        i + 1, from_pos[1])]
                # move the piece down
                self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
        return acceptable


class TrainingGame(Game):
    def __init__(self, board_state: np.array = None) -> None:
        super().__init__()
        if board_state is not None: self._board = deepcopy(board_state)

    def play(self, player1: Player, player2: Player, board_state: np.array = None, next_to_move: int = 0) -> int:
        '''Play the game. Returns the winning player'''
        if board_state is not None: self._board = deepcopy(board_state)
        players = [player1, player2]
        current_player_idx = 1 + next_to_move
        winner = -1
        while winner < 0:
            # get board old state
            old_state = self.get_board()
            current_player_idx += 1
            current_player_idx %= len(players)
            ok = False
            while not ok:
              from_pos, slide = players[current_player_idx].make_move(self)
              ok = self.move(from_pos, slide, current_player_idx)
            #my player is in exploration
            if isinstance(players[current_player_idx], MyPlayer) and players[current_player_idx].Is == PureyaIs.EXPLORING:
                board_copy = self.get_board()
                # #if the state is missing, I add it (no actions still) USELESS 'CAUSE I HAVE NO ACTIONS STARTIN FROM IT!
                # if tuple(board_copy.ravel()) not in players[current_player_idx].model:
                #     # get my model action
                #     players[current_player_idx].model[tuple(board_copy.ravel())] = []
                #if the starting state is missing I add it too
                if tuple(old_state.ravel()) not in players[current_player_idx].model:
                    players[current_player_idx].model[tuple(old_state.ravel())] = []
                #if the action is a new one, I add it
                list_actions = players[current_player_idx].model[tuple(old_state.ravel())]
                if all([(from_pos, slide) != (existing_from_pos, existing_move)
                       for _, existing_from_pos, existing_move, _, _ in list_actions]):
                    list_actions.append((board_copy, from_pos, slide, 0, current_player_idx))
            winner = self.check_winner()

        return winner

    def move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id > 2:
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.__take((from_pos[1], from_pos[0]), player_id)
        if acceptable:
            acceptable = self.__slide((from_pos[1], from_pos[0]), slide)
            if not acceptable:
                self._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable

    def __take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        '''Take piece'''
        # acceptable only if in border
        acceptable: bool = (
            # check if it is in the first row
            (from_pos[0] == 0 and from_pos[1] < 5)
            # check if it is in the last row
            or (from_pos[0] == 4 and from_pos[1] < 5)
            # check if it is in the first column
            or (from_pos[1] == 0 and from_pos[0] < 5)
            # check if it is in the last column
            or (from_pos[1] == 4 and from_pos[0] < 5)
            # and check if the piece can be moved by the current player
        ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
        if acceptable:
            self._board[from_pos] = player_id
        return acceptable

    def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        '''Slide the other pieces'''
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == 4 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == 4 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == (0, 0) and (
                slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == (4, 0) and (
                slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == (0, 4) and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        # if it is
        if acceptable:
            # take the piece
            piece = self._board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                self._board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], self._board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    self._board[(i, from_pos[1])] = self._board[(
                        i - 1, from_pos[1])]
                # move the piece up
                self._board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], self._board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    self._board[(i, from_pos[1])] = self._board[(
                        i + 1, from_pos[1])]
                # move the piece down
                self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
        return acceptable


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class MyPlayer_dqn(Player):
    def __init__(self, trained: bool = False, model_path: list[str] = [], save: bool = False) -> None:
        super().__init__()
        #for the sake of defining stuff
        self.last_status = None
        self.replay_mem0 = []
        self.replay_mem1 = []
        self.REPLAY_MEM_SIZE = 10_000
        self.exploration_rate = 0

        # if I want to use a trained model
        if trained:
            # if I want to load a pretrained model
            if len(model_path) > 0:
                self.model = MyMLP(25, 128, 16, 4)
                self.model.load_state_dict(torch.load(model_path[0]))
                #self.model1 = torch.load(model_path[1])
            else:
                self.model = None
                #self.model1 = None
                self.train()

                if save:
                    torch.save(self.model.state_dict(), "my_model0.pth")
                    #torch.save(self.model1.state_dict(), "my_model1.pth")
        # if I don't want to use a trained model
        else:
            self.make_move = self.make_move_random

    def compute_reward(self, player_idx, old_state, new_state, is_winner, penalty=False):
        if is_winner:
            return 1
        # this if I make a not allowed move
        elif penalty:
            return -1
        else:
            return -1

    def get_experience(self, old_state, action1, action2, reward, new_state, is_terminal, player_pos):
        old_state = tuple(old_state.ravel())
        new_state = tuple(new_state.ravel())
        key = (old_state, new_state)
        if player_pos == 0:
            self.replay_mem0[key] = tuple([old_state, action1, action2, reward, new_state, is_terminal])

            while len(self.replay_mem0) >= self.REPLAY_MEM_SIZE:
                oldest_key = next(iter(self.replay_mem0))
                del self.replay_mem0[oldest_key]
        else:
            self.replay_mem1[key] = tuple([old_state, action1, action2, reward, new_state, is_terminal])

            while len(self.replay_mem1) >= self.REPLAY_MEM_SIZE:
                oldest_key = next(iter(self.replay_mem1))
                del self.replay_mem1[oldest_key]

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        #I preprocess my input by flattening the state
        board_copy = game.get_board()
        #print(board_copy)
        board_copy = board_copy.ravel()
        #print("bbbb", board_copy)

        #if I have already tried my deterministic choice
        # and the state is the same, it wasn't allowed so I'll go random or get stuck
        if np.all(self.last_status == board_copy):
            return self.make_move_random(game)
        self.last_status = board_copy

        if random.random() > self.exploration_rate:
            model_input = torch.tensor(board_copy, dtype=torch.float32).unsqueeze(0)
            #if game.get_current_player() == 0:
            from_pos = torch.argmax(self.model.forward_action1(model_input), dim=1).item()
            move = torch.argmax(self.model.forward_action2(model_input), dim=1).item()
            #else:
            #    from_pos = torch.argmax(self.model1.forward_action1(model_input), dim=1).item()
             #   move = torch.argmax(self.model1.forward_action2(model_input), dim=1).item()
        else:
            return self.make_move_random(game)
        #print("F:", from_pos, "M:", move)

        #bring the format back to the one of the game
        if from_pos < 5:
            from_pos = tuple([0, from_pos])
        elif 4 < from_pos < 8:
            from_pos = tuple([from_pos - 4, 4])
        elif 7 < from_pos < 13:
            from_pos = tuple([4, 12 - from_pos])
        else:
            from_pos = tuple([16 - from_pos, 0])
        #print("F:",from_pos, "M:",move)
        return from_pos, Move(move)

    def get_pos(self, from_pos):
        pos_x, pos_y = from_pos
        if pos_y == 0:
            return pos_x
        elif pos_y == 4:
            return 12 - pos_x
        elif pos_x == 0:
            return 16 - pos_y
        else:
            return 4 + pos_y

    def make_move_random(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        #from_pos = (random.randint(0, 4), random.randint(0, 4))
        #move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])

        myboard = game.get_board()
        if 12 > sum([ sum([1 for y in range(myboard.shape[0]) if myboard[x, y] == -1]) for x in range(myboard.shape[0])]):
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        else:
            from_pos = None
            while from_pos is None or myboard[from_pos[1], from_pos[0]] != -1:
                from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        #print("F:", from_pos, "M:", move)
        return from_pos, move

    def train(self):
        self.model = MyMLP(25, 128, 16, 4)
        #self.model1 = MyMLP(25, 180, 16, 4)

        # 4 the sake of visualization
        wins = []
        losses = []
        # put the model in training mode
        self.model.train()
        # self.model1.train()

        # setup the training
        DISCOUNT_RATE = 0.99
        NUM_EPISODES = 50000
        TARGET_UP_RATE = 50
        MAX_EX_RATE = 1
        MIN_EX_RATE = 0.001
        EX_RATE_DECAY = 0.0005
        self.exploration_rate = MAX_EX_RATE

        # accessory data structure for training
        self.replay_mem0 = {}
        self.replay_mem1 = {}
        tar_model = deepcopy(self.model)

        # define the optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for episode in tqdm(range(NUM_EPISODES)):
            mygame = TrainingGame_dqn()
            # ep_model = None
            # tar_model = None
            ep_model = self.model
            plays1st = episode % 2 == 0
            if not plays1st:
                # self.model1.train()
                # self.model.eval()
                winner = mygame.play(RandomPlayer(), self)
                # ep_model = self.model1
                # tar_model = self.model
            else:
                # self.model.train()
                # self.model1.eval()
                winner = mygame.play(self, RandomPlayer())
                # ep_model = self.model
                # tar_model = self.model1
            wins.append(episode % 2 != winner)

            # define the optimizer
            # L_R = 0.0001 if episode > 16000 else 0.001
            # optimizer = optim.SGD(ep_model.parameters(), lr=L_R, momentum=0.9)

            # get a batch from replay memory and update the model
            if plays1st:
                random_keys = random.choices(list(self.replay_mem0.keys()), k=256)
            else:
                random_keys = random.choices(list(self.replay_mem1.keys()), k=256)
            states, actions1, actions2, rewards, next_states, dones = [], [], [], [], [], []
            for key in random_keys:
                if plays1st:
                    random_obs = self.replay_mem0[key]
                else:
                    random_obs = self.replay_mem1[key]
                states.append(random_obs[0])
                actions1.append(random_obs[1])
                actions2.append(random_obs[2].value)
                rewards.append(random_obs[3])
                next_states.append(random_obs[4])
                dones.append(random_obs[5])
            states = torch.tensor(states, dtype=torch.float32)
            actions1 = torch.tensor(actions1, dtype=torch.long)
            actions2 = torch.tensor(actions2, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            #next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # q values of the current state s_i
            q_values_a1 = ep_model.forward_action1(states)
            q_values_a2 = ep_model.forward_action2(states)

            # uses a1_i and a2_i to get just the q values of interest
            q_values_selected_a1 = q_values_a1[torch.arange(q_values_a1.size(0)), actions1]
            q_values_selected_a2 = q_values_a2[torch.arange(q_values_a2.size(0)), actions2]

            #simulate the opponent move of each next state:
            target_states = []
            for nxst in next_states:
                mygame = TrainingGame_dqn(np.array(nxst).reshape((5,5)))
                ok = False
                # do it in a simulated way
                while not ok:
                    # if random.random() > 0.3:
                    #     model_input = torch.tensor(nxst, dtype=torch.float32).unsqueeze(0)
                    #     # if game.get_current_player() == 0:
                    #     from_pos = torch.argmax(self.model.forward_action1(model_input), dim=1).item()
                    #     move = torch.argmax(self.model.forward_action2(model_input), dim=1).item()
                    #     # bring the format back to the one of the game
                    #     if from_pos < 5:
                    #         from_pos = tuple([0, from_pos])
                    #     elif 4 < from_pos < 8:
                    #         from_pos = tuple([from_pos - 4, 4])
                    #     elif 7 < from_pos < 13:
                    #         from_pos = tuple([4, 12 - from_pos])
                    #     else:
                    #         from_pos = tuple([16 - from_pos, 0])
                    #     slide = Move(move)
                    # else:
                    #     from_pos, slide = self.make_move_random(mygame)
                    from_pos, slide = self.make_move_random(mygame)

                    ok = mygame.move(from_pos, slide, 0 if plays1st else 1)
                target_states.append(mygame.get_board().ravel())
            #get my target state
            target_states = torch.tensor(target_states, dtype=torch.float32)

            # target q values using the Bellman equation
            target_q_values_next_states_a1 = tar_model.forward_action1(target_states)
            target_q_values_next_states_a2 = tar_model.forward_action2(target_states)

            # select the maximum q value for each action in the next state
            max_q_values_next_states_a1 = target_q_values_next_states_a1.mean(1).detach()
            max_q_values_next_states_a2 = target_q_values_next_states_a2.mean(1).detach()

            # get target q values using the Bellman equation
            target_q_values_a1 = rewards + (1 - dones) * DISCOUNT_RATE * +max_q_values_next_states_a1
            target_q_values_a2 = rewards + (1 - dones) * DISCOUNT_RATE * +max_q_values_next_states_a2

            # the Huber loss of each action
            loss_a1 = F.smooth_l1_loss(q_values_selected_a1, target_q_values_a1)
            loss_a2 = F.smooth_l1_loss(q_values_selected_a2, target_q_values_a2)

            # print("tqv1",target_q_values_a1,"rwd",rewards,"qnext1",max_q_values_next_states_a1,
            #       "tqv2", target_q_values_a2, "rwd", rewards, "qnext2", max_q_values_next_states_a2,
            #       "losses:",loss_a1,"-",loss_a2)

            # the total loss is the bare sum of the single sums
            total_loss = loss_a1 + loss_a2
            losses.append(total_loss.detach().numpy())

            # update the model
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # once in a while I update the target network
            if episode % TARGET_UP_RATE == 0:
                tar_model = deepcopy(self.model)

            # update the exploration rate
            self.exploration_rate = MIN_EX_RATE + (MAX_EX_RATE - MIN_EX_RATE) * np.exp(-EX_RATE_DECAY * episode)

        # plot the training : actually wins + draws
        win_size = 1000
        # avg_wins = [sum(wins[i: i + win_size]) / win_size
        #             for i, _ in enumerate(wins[win_size - 1:])]
        avg_wins = [sum(wins[i * win_size: (i + 1) * win_size]) / win_size
                    for i in range(int(len(wins) / win_size))]
        avg_losses = [sum(losses[i * win_size: (i + 1) * win_size]) / win_size
                      for i in range(int(len(losses) / win_size))]
        plt.plot(range(len(avg_wins)), avg_wins)
        #plt.plot(range(len(avg_losses)), avg_losses)
        plt.show()
        # post training stuff

        # put the model in evaluation mode
        self.model.eval()
        # self.model1.eval()


class PureyaIs(Enum):
    EXPLORING = 0,
    ESTIMATING = 1,
    GAMING = 2,
    FAST_EST = 3


class MyPlayer(Player):
    def __init__(self, trained: bool = False, model_path: list[str] = [], save: bool = False) -> None:
        super().__init__()
        #for the sake of defining stuff
        self.last_status = None
        self.Is = PureyaIs.GAMING
        self.exploration_rate = 0
        self.fast_est_games = 60
        if trained:
            # if I want to load a pretrained model
            if len(model_path) > 0:
                with open(model_path[0], 'rb') as f:
                    self.model = pickle.load(f)
            else:
                self.model = {}
                self.train(num_games_expl=10, num_games_est=30, num_games_finetuning=100, num_epochs=10)
                #self.train(num_games_expl=100, num_games_est=50, num_games_finetuning=100, num_epochs=10)

                if save:
                    with open(f'my_model.pkl', 'wb') as f:
                        pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
        # if I don't want to use a trained model
        else:
            self.make_move = self.make_move_random

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        #I preprocess my input by flattening the state
        board_cp = game.get_board()
        board_copy = board_cp.ravel()

        #if I have already tried my deterministic choice
        # and the state is the same, it wasn't allowed so I'll go random or get stuck
        if np.all(self.last_status == board_copy):
            return self.make_move_random(game)
        self.last_status = board_copy

        if self.Is == PureyaIs.EXPLORING and random.random() > self.exploration_rate:
            return self.make_move_random(game)
        elif self.Is == PureyaIs.GAMING:
            if tuple(board_copy) in self.model and len(self.model[tuple(board_copy)]) > 1:
                # get my model action
                list_moves = self.model[tuple(board_copy)]
                _, from_pos, move, _, _ = max(list_moves, key=lambda x: x[3])
                return from_pos, move
            # if I can't find it , I'll go with fast estimates
            else:
                self.Is = PureyaIs.FAST_EST
                list_board_cp_actions = []
                for ep in range(self.fast_est_games):
                    #get new random allowed action
                    mygame = TrainingGame(game.get_board())
                    ok = False
                    # do it in a simulated way
                    while not ok:
                        from_pos, slide = self.make_move(mygame)
                        ok = mygame.move(from_pos, slide, game.get_current_player())
                    #opponent starts whatever its position is
                    if 1 - game.get_current_player() == 0:
                        winner = mygame.play(RandomPlayer(), self, board_state=mygame.get_board())
                    else:
                        winner = mygame.play(self, RandomPlayer(), board_state=mygame.get_board(), next_to_move=1)

                    if game.get_current_player() == winner:
                        # if the tuple exists, I update it, I add it otherwise
                        tuple_index = next(
                            (i for i, (existing_from_pos, existing_move, _) in enumerate(list_board_cp_actions)
                             if (existing_from_pos, existing_move) == (from_pos, slide)), None)
                        if tuple_index is not None:
                            list_board_cp_actions[tuple_index] = (*list_board_cp_actions[tuple_index][:2],
                                                                  list_board_cp_actions[tuple_index][2] + 1)
                        else:
                            list_board_cp_actions.append((from_pos, slide, 1))

                self.Is = PureyaIs.GAMING
                fast_from_pos, fast_slide, _ = max(list_board_cp_actions, key=lambda x: x[2])
                return fast_from_pos, fast_slide
        #exploring following policy or estimating or fast estimating
        else:
            if tuple(board_copy) in self.model:
                # get my model action
                list_moves = self.model[tuple(board_copy)]
                if len(list_moves) > 1:
                    _, from_pos, move, _, _ = max(list_moves, key=lambda x: x[3])
                    return  from_pos, move
                else:
                    return self.make_move_random(game)
            # if I can't find it, I'll go random
            else:
                return self.make_move_random(game)

    def make_move_random(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        #from_pos = (random.randint(0, 4), random.randint(0, 4))
        #move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])

        myboard = game.get_board()
        if 12 > sum([ sum([1 for y in range(myboard.shape[0]) if myboard[x, y] == -1]) for x in range(myboard.shape[0])]):
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        else:
            from_pos = None
            while from_pos is None or myboard[from_pos[1], from_pos[0]] != -1:
                from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        #print("F:", from_pos, "M:", move)
        return from_pos, move

    def train(self, num_games_expl:int, num_games_est:int, num_games_finetuning:int, num_epochs:int):
        MAX_EX_RATE = 1
        MIN_EX_RATE = 0.001
        EX_RATE_DECAY = 0.01
        self.exploration_rate = MAX_EX_RATE
        self.model = {}
        init_state = tuple((np.ones((5, 5), dtype=np.uint8) * -1).ravel())
        self.model[init_state] = []

        #for n epochs
        outer_bar = tqdm(total=num_epochs, desc="TRAINING EPOCHS - ", position=0)
        for ep in range(num_epochs):
            #1 use the states of my model to get random plays and add new states
            self.Is = PureyaIs.EXPLORING
            for expl in range(num_games_expl):
                mygame = TrainingGame()
                if expl % 2 == 0:
                    winner = mygame.play(RandomPlayer(), self)
                else:
                    winner = mygame.play(self, RandomPlayer())
            #print #states
            #outer_bar.write(f"epoch:{ep + 1} #states:{len(self.model)}")

            #2 for each state, from the terminal to the initial ones
            num_games = num_games_finetuning if ep == num_epochs - 1 else num_games_est
            sorted_keys = sorted(self.model.keys(), key=lambda x: x.count(-1))
            self.Is = PureyaIs.ESTIMATING
            inner_bar = tqdm(total=len(sorted_keys), desc=f"ESTIMATIONS - ", position=1)
            for state_key in sorted_keys:
                #if I have more than one action
                if len(self.model[state_key]) > 1:
                    list_actions = self.model[state_key]
                    #for each of their actions
                    for act in list_actions:
                        my_board, _, _, _, p_pos = act
                        winrate = 0
                        #do random plays
                        for est in range(num_games):
                            mygame = TrainingGame()
                            if p_pos == 1:
                                winner = mygame.play(RandomPlayer(), self, board_state=my_board)
                            else:
                                winner = mygame.play(self, RandomPlayer(), board_state=my_board, next_to_move=1)
                            #if it wins I increase the winrate by 1
                            if p_pos != winner:
                                winrate += 1
                        act = (*act[:3], winrate, act[4])
                inner_bar.update(1)
            inner_bar.close()

            # update the exploration rate
            self.exploration_rate = MIN_EX_RATE + (MAX_EX_RATE - MIN_EX_RATE) * np.exp(-EX_RATE_DECAY * ep)
            outer_bar.update(1)
        outer_bar.close()

        # filter each state with just one action
        keys_to_remove = [key for key, value in self.model.items() if len(value) <= 1]
        for key in keys_to_remove:
            del self.model[key]

        self.Is = PureyaIs.GAMING

if __name__ == '__main__':
    g = Game()
    #player1 = MyPlayer(trained=True, save=True)
    #player1 = MyPlayer(trained=True, model_path=["my_model.pkl"])
    player1 = MyPlayer(trained=True, model_path=["my_model_90_90.pkl"])#more stable result so far...
    #player1 = MyPlayer_dqn(trained=True, save=True)
    #player1 = MyPlayer_dqn(trained=True, model_path=["my_model0_75.pth"])
    player2 = RandomPlayer()

    #print("rep_mem_size:", len(player1.replay_mem0), " + ", len(player1.replay_mem1))
    # key_n = 0
    # ones_n = 0
    # for key in player1.model.keys():
    #     key_n += 1
    #     if len(player1.model[key]) > 1: print("key:",key_n, " act:", len(player1.model[key]))
    #     if len(player1.model[key]) == 1: ones_n += 1
    # print("total ones:", ones_n)

    ########################################################
    win1 = 0
    draw1 = 0
    plays = 1000
    for i in tqdm(range(plays)):
        g = Game()
        winner = g.play(player1, player2)
        if winner == 0:
            win1 += 1
        elif winner == -2:
            draw1 += 1
    print("Win perc as 1st:", win1 * 100 / plays)
    print("Draw perc as 1st:", draw1 * 100 / plays)
    ########################################################
    win2 = 0
    draw2 = 0
    plays = 1000
    for i in tqdm(range(plays)):
        g = Game()
        winner = g.play(player2, player1)
        if winner == 1:
            win2 += 1
        elif winner == -2:
            draw2 += 1
    print("Win perc as 2nd:", win2 * 100 / plays)
    print("Draw perc as 2nd:", draw2 * 100 / plays)

