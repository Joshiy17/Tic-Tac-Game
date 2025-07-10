import random

class QLearningTicTacToe:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def get_state(self, board):
        return ''.join(board)

    def choose_action(self, state, available_moves):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)
        q_values = [self.q_table.get((state, move), 0) for move in available_moves]
        max_q = max(q_values)
        return available_moves[q_values.index(max_q)]

    def learn(self, state, action, reward, next_state, done, next_moves):
        current_q = self.q_table.get((state, action), 0)
        if done:
            self.q_table[(state, action)] = reward
        else:
            next_q = max([self.q_table.get((next_state, a), 0) for a in next_moves], default=0)
            new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
            self.q_table[(state, action)] = new_q

    def check_winner(self, board, player):
        wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                (0, 3, 6), (1, 4, 7), (2, 5, 8),
                (0, 4, 8), (2, 4, 6)]
        for i, j, k in wins:
            if board[i] == board[j] == board[k] == player:
                return True
        return False

    def train(self, episodes=5000):
        for _ in range(episodes):
            board = [' '] * 9
            done = False
            state = self.get_state(board)

            while not done:
                available = [i for i in range(9) if board[i] == ' ']
                action = self.choose_action(state, available)
                board[action] = 'X'

                if self.check_winner(board, 'X'):
                    self.learn(state, action, 1, None, True, [])
                    break
                elif ' ' not in board:
                    self.learn(state, action, 0.5, None, True, [])
                    break

                # Random opponent move
                opp_move = random.choice([i for i in range(9) if board[i] == ' '])
                board[opp_move] = 'O'
                if self.check_winner(board, 'O'):
                    self.learn(state, action, -1, None, True, [])
                    break

                next_state = self.get_state(board)
                next_available = [i for i in range(9) if board[i] == ' ']
                self.learn(state, action, 0, next_state, False, next_available)
                state = next_state

    def print_board(self, board):
        print()
        print(f" {board[0]} | {board[1]} | {board[2]}")
        print("---+---+---")
        print(f" {board[3]} | {board[4]} | {board[5]}")
        print("---+---+---")
        print(f" {board[6]} | {board[7]} | {board[8]}")
        print()

    def play(self):
        print("\nYou are playing against the trained AI!")
        board = [' '] * 9
        while True:
            self.print_board(board)
            # Human turn
            try:
                user_move = int(input("Your move (0–8): "))
                while board[user_move] != ' ':
                    user_move = int(input("Invalid! Choose again (0–8): "))
            except:
                print("Invalid input. Please enter 0–8.")
                continue

            board[user_move] = 'O'
            if self.check_winner(board, 'O'):
                self.print_board(board)
                print("You win! 🎉")
                break
            if ' ' not in board:
                self.print_board(board)
                print("It's a draw!")
                break

            # AI move
            state = self.get_state(board)
            ai_move = self.choose_action(state, [i for i in range(9) if board[i] == ' '])
            board[ai_move] = 'X'
            print(f"AI played at position {ai_move}")
            if self.check_winner(board, 'X'):
                self.print_board(board)
                print("AI wins! 🤖")
                break
            if ' ' not in board:
                self.print_board(board)
                print("It's a draw!")
                break

# Run in PyCharm
if __name__ == "__main__":
    game = QLearningTicTacToe()
    game.train(5000)  # Train for 5,000 games
    game.play()       # Play against you
