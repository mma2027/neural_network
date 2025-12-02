# mcts.py

import math

from helper import (
    ROWS, COLS,
    get_valid_moves,
    drop_piece,
    check_win,
    is_draw,
    board_to_vector,
)


class MCTSNode:
    def __init__(self, prior: float):
        # Prior probability from policy network
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
        model: your neural net with .forward_policy_value(x)
        cpuct: exploration constant (like AlphaZero's c_puct)
        """
        self.model = model
        self.cpuct = cpuct

    def _terminal_value(self, board, player_to_move: int):
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
        return 0.0  # draw

    def _nn_policy_value(self, board, player_to_move: int):
        """
        Use the NN to get (policy, value) for player_to_move.
        policy: length-7 probs, value in [-1,1].
        """
        x = board_to_vector(board, player_to_move)
        pi, v_prob = self.model.forward_policy_value(x)
        # map [0,1] -> [-1,1]
        v = 2.0 * float(v_prob) - 1.0
        return pi, v

    def _masked_priors(self, board, player_to_move: int):
        """
        Get NN policy and mask invalid moves.
        Returns dict: move -> prior.
        """
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            return {}

        pi_raw, _ = self._nn_policy_value(board, player_to_move)
        pi_raw = list(pi_raw)

        priors = {}
        s = 0.0
        for c in valid_moves:
            p = max(pi_raw[c], 1e-6)  # avoid zero
            priors[c] = p
            s += p

        if s <= 0.0 or not math.isfinite(s):
            # fallback uniform
            uniform = 1.0 / len(valid_moves)
            return {c: uniform for c in valid_moves}

        for c in priors:
            priors[c] /= s

        return priors

    def _select_child(self, node: MCTSNode):
        """
        Select a child using PUCT.
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
        One MCTS simulation.
        Returns value in [-1,1] from the perspective of player_to_move
        at this node.
        """
        # 1. Terminal?
        terminal = self._terminal_value(board, player_to_move)
        if terminal is not None:
            return terminal

        # 2. Expand if leaf
        if not node.children:
            priors = self._masked_priors(board, player_to_move)
            if not priors:
                v = 0.0
            else:
                for m, p in priors.items():
                    node.children[m] = MCTSNode(prior=p)
                # Value for current state
                _, v = self._nn_policy_value(board, player_to_move)

            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            return v

        # 3. Otherwise, select a child
        move, child = self._select_child(node)

        # 4. Apply move
        drop_piece(board, move, player_to_move)
        next_player = 2 if player_to_move == 1 else 1

        # 5. Recurse with sign flip (perspective changes)
        v = -self._simulate(board, next_player, child)

        # 6. Backprop
        node.N += 1
        node.W += v
        node.Q = node.W / node.N

        return v

    def search(self, board, player_to_move: int, num_simulations: int = 200):
        """
        Run MCTS and return best move.
        """
        root = MCTSNode(prior=1.0)

        for _ in range(num_simulations):
            b = [row[:] for row in board]
            self._simulate(b, player_to_move, root)

        if not root.children:
            return -1

        best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        return best_move

    def search_with_policy(self, board, player_to_move: int, num_simulations: int = 200):
        """
        Run MCTS and return (best_move, visit_policy),
        where visit_policy is a length-7 vector of visit-count-based probs.
        """
        root = MCTSNode(prior=1.0)

        for _ in range(num_simulations):
            b = [row[:] for row in board]
            self._simulate(b, player_to_move, root)

        if not root.children:
            return -1, [0.0] * COLS

        # Best move by visit count
        best_move, _ = max(root.children.items(), key=lambda kv: kv[1].N)

        # Build policy from visit counts
        visits = [0.0] * COLS
        total_N = 0.0
        for m, child in root.children.items():
            visits[m] = child.N
            total_N += child.N

        if total_N <= 0.0:
            # fallback: one-hot on best_move
            pi = [0.0] * COLS
            if 0 <= best_move < COLS:
                pi[best_move] = 1.0
            return best_move, pi

        pi = [v / total_N for v in visits]
        return best_move, pi


def mcts_move(board, player: int, model, num_simulations: int = 200, cpuct: float = 1.4) -> int:
    """
    Original API: just return the move.
    """
    mcts = MCTS(model=model, cpuct=cpuct)
    return mcts.search(board, player_to_move=player, num_simulations=num_simulations)


def mcts_move_with_policy(board, player: int, model, num_simulations: int = 200, cpuct: float = 1.4):
    """
    New API: return (move, visit_policy).
    """
    mcts = MCTS(model=model, cpuct=cpuct)
    return mcts.search_with_policy(board, player_to_move=player, num_simulations=num_simulations)
