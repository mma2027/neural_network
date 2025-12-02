# mcts.py

import math

from helper import (
    get_valid_moves,
    drop_piece,
    check_win,
    is_draw,
    board_to_vector,
)


class MCTSNode:
    def __init__(self, prior: float):
        # Prior probability from "policy" (we now derive from NN, not uniform)
        self.P = prior

        # Visit statistics
        self.N = 0          # visit count
        self.W = 0.0        # total value
        self.Q = 0.0        # mean value

        # Children: move -> MCTSNode
        self.children = {}


class MCTS:
    def __init__(self, model, cpuct: float = 1.4):
        """
        model: your neural net with .forward(x) -> [0,1] win prob
        cpuct: exploration constant (like AlphaZero's c_puct)
        """
        self.model = model
        self.cpuct = cpuct

    def _terminal_value(self, board, player_to_move: int) -> float | None:
        """
        Check if board is terminal. If so, return value in [-1, 0, 1]
        from the perspective of player_to_move.
        If not terminal, return None.
        """
        p1_wins = check_win(board, 1)
        p2_wins = check_win(board, 2)
        draw = is_draw(board)

        if not (p1_wins or p2_wins or draw):
            return None

        if p1_wins:
            return 1.0 if player_to_move == 1 else -1.0
        if p2_wins:
            return 1.0 if player_to_move == 2 else -1.0
        # draw
        return 0.0

    def _nn_value(self, board, player_to_move: int) -> float:
        """
        Use the neural network to evaluate the position for player_to_move.
        Returns a value in [-1,1] (converted from [0,1] sigmoid).
        """
        x = board_to_vector(board, player_to_move)
        p = float(self.model.forward(x))  # [0,1] win prob
        return 2.0 * p - 1.0             # map [0,1] -> [-1,1]

    def _policy_priors(self, board, player_to_move: int, moves):
        """
        Build policy priors over legal moves using the NN.

        For each move, we:
          - apply it to a copy of the board,
          - evaluate the resulting position for player_to_move,
          - use those evaluations as logits, then softmax.

        This gives us non-uniform priors that favor moves that
        look more promising to the NN.
        """
        if not moves:
            return {}

        # Evaluate each child state
        logits = []
        for m in moves:
            # copy board
            b_child = [row[:] for row in board]
            drop_piece(b_child, m, player_to_move)
            # evaluate from player_to_move's perspective
            v_child = self._nn_value(b_child, player_to_move)
            logits.append(v_child)

        # Softmax over logits for numerical stability
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        sum_exps = sum(exps)

        # Fallback to uniform if something goes weird
        if sum_exps <= 0.0 or not math.isfinite(sum_exps):
            prior = 1.0 / len(moves)
            return {m: prior for m in moves}

        priors = {}
        for m, e in zip(moves, exps):
            priors[m] = e / sum_exps
        return priors

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """
        Select a child using the PUCT (AlphaZero-style) formula.
        Returns (move, child_node).
        """
        best_score = -1e9
        best_move = None
        best_child = None

        parent_visits = node.N + 1e-8

        for move, child in node.children.items():
            u = child.Q + self.cpuct * child.P * math.sqrt(parent_visits) / (1.0 + child.N)
            if u > best_score:
                best_score = u
                best_move = move
                best_child = child

        return best_move, best_child

    def _simulate(self, board, player_to_move: int, node: MCTSNode) -> float:
        """
        Run one MCTS simulation from (board, player_to_move, node).
        Returns value in [-1,1] from the perspective of player_to_move
        at *this* node.
        """
        # 1. Check for terminal state
        terminal = self._terminal_value(board, player_to_move)
        if terminal is not None:
            return terminal

        # 2. If this node has no children yet, expand and evaluate with NN
        if not node.children:
            moves = get_valid_moves(board)
            if not moves:
                v = 0.0  # no moves: treat as draw
            else:
                # --- NEW: use NN-derived priors instead of uniform ---
                priors = self._policy_priors(board, player_to_move, moves)
                for m in moves:
                    node.children[m] = MCTSNode(prior=priors.get(m, 0.0))

                # NN value of the *current* state
                v = self._nn_value(board, player_to_move)

            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            return v

        # 3. Otherwise, select a child using PUCT
        move, child = self._select_child(node)

        # 4. Apply the move to the board
        drop_piece(board, move, player_to_move)
        next_player = 2 if player_to_move == 1 else 1

        # 5. Recurse; sign flip because perspective changes
        v = -self._simulate(board, next_player, child)

        # 6. Backpropagate
        node.N += 1
        node.W += v
        node.Q = node.W / node.N

        return v

    def search(self, board, player_to_move: int, num_simulations: int = 200) -> int:
        """
        Run MCTS from (board, player_to_move) and return the best move.
        Best move = child with highest visit count N.
        """
        root = MCTSNode(prior=1.0)

        for _ in range(num_simulations):
            b = [row[:] for row in board]  # fresh copy
            self._simulate(b, player_to_move, root)

        if not root.children:
            # No moves (terminal), return -1 as a sentinel
            return -1

        # Choose move with the highest visit count
        best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        return best_move


def mcts_move(board, player: int, model, num_simulations: int = 200, cpuct: float = 1.4) -> int:
    """
    Convenience wrapper: pick a move for `player` on `board` using MCTS+NN.
    """
    mcts = MCTS(model=model, cpuct=cpuct)
    return mcts.search(board, player_to_move=player, num_simulations=num_simulations)
