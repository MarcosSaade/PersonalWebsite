import React from 'react';
import PageTemplate from '../components/PageTemplate';
import chessImage from '../images/chess.webp';
import ui from '../images/chess/ui.png';

export default function ChessPage() {
  return (
    <PageTemplate title="Building a Chess Engine with Minimax and Alpha-Beta Pruning" image={chessImage}>
      <p>
        I love chess. It’s beautiful, ancient, and intellectually satisfying—everything I like in a game. So one day, I set out to build a chess engine from scratch. Just Python and Pygame for the UI. The result is a fully functional engine that supports all standard rules (castling, checkmate, promotion), a Minimax-based AI with Alpha-Beta pruning, and even a small opening book. I built this the summer after high school, as a side project. I learned a lot about algorithms, data structures, and it even improved my own chess skills!. Here’s how I did it.
      </p>

      <h2>Board Representation</h2>
      <p>
        The board is represented as an 8×8 2D list of strings like <code>"wP"</code> (white pawn), <code>"bQ"</code> (black queen), or <code>"  "</code> for empty squares. This kept things simple and readable during move generation and evaluation.
      </p>

      <pre><code className="language-js">{`[
  ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'],
  ['bP', 'bP', 'bP', 'bP', 'bP', 'bP', 'bP', 'bP'],
  ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '],
  ...
  ['wP', 'wP', 'wP', 'wP', 'wP', 'wP', 'wP', 'wP'],
  ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR'],
]`}</code></pre>

      <h2>Move Generation</h2>
      <p>
        For each piece type (pawn, knight, bishop, etc), I implemented custom logic to generate all valid moves. The result is a <code>get_moves()</code> function that scans the board and returns a list of legal moves for the given color.
      </p>

      <pre><code className="language-python">{`def get_moves(board, color):
    moves = []
    for i in range(8):
        for j in range(8):
            if board[i][j][0] == color:
                piece = board[i][j][1]
                if piece == 'N':
                    moves += knight_moves(board, color)
                elif piece == 'B':
                    moves += bishop_moves(board, color)
                # etc...
    return remove_duplicates(moves)`}</code></pre>

      <p>
        I also wrote logic to check for obstacles (for rooks, bishops, queens), pawn promotions, and castling rights. This was tricky but satisfying.
      </p>


      <h2>Minimax and Alpha-Beta Pruning</h2>
      <p>
        Minimax is a recursive decision-making algorithm used in two-player games. One player tries to maximize the score (Max), while the opponent tries to minimize it (Min). The idea is to simulate every possible sequence of moves up to a certain depth, and evaluate the leaf positions using a scoring function. Then, these scores are propagated back up the tree to choose the optimal move.
      </p>

      <p>
        For example, if it's my turn and I can choose between three moves, I simulate all my opponent's responses to each move, then all my possible replies to those responses, and so on. At the leaves, I use an evaluation function to assign a numeric value to the board. I then assume my opponent will play optimally and choose the line that minimizes my score, and I counter that by choosing the line that gives me the best possible worst-case outcome.
      </p>

      <p>
        The problem is that the number of positions grows exponentially with depth. To address that, I used Alpha-Beta pruning. It keeps track of two values: alpha (the best already explored score for Max) and beta (the best for Min). If we find that a move can't possibly improve the outcome for the current player, we stop searching down that path. This drastically cuts down the number of evaluations.
      </p>

      <p>
        Here’s the core implementation:
      </p>

      <pre><code className="language-python">{`def minimax(board, depth, is_maximizing, color, alpha, beta):
    if depth == 0:
        return static_eval(board), None

    moves = order_moves(board, get_moves(board, color))
    best_move = None

    for move in moves:
        new_board = make_move(board, move)
        score, _ = minimax(new_board, depth - 1, not is_maximizing, next_color, alpha, beta)

        if is_maximizing:
            if score > alpha:
                alpha = score
                best_move = move
        else:
            if score < beta:
                beta = score
                best_move = move

        if beta <= alpha:
            break

    return (alpha, best_move) if is_maximizing else (beta, best_move)`}</code></pre>

      <h2>Evaluation Function</h2>
      <p>
        The evaluation function combines material count with heuristics like:
      </p>
      <ul>
        <li>Piece development (early game incentives to centralize)</li>
        <li>Pawn structure (penalizing doubled or isolated pawns)</li>
        <li>Castling (increased score for castled kings)</li>
        <li>Rooks on open files</li>
        <li>King safety (rudimentary heuristics)</li>
      </ul>

      <p>For example:</p>

      <pre><code className="language-python">{`if piece == 'wN' and position in center:
    score += 0.05
elif piece == 'wP' and col in [3, 4]:
    score += 0.2  # Central pawns`}</code></pre>

      <p>Of course, a better approach would be to learn the evaluation function using a neural network, but I wanted to keep it simple for now. </p>

      <h2>Opening Book</h2>
      <p>
        I hardcoded a small opening book with popular lines (e.g., Sicilian, Italian Game, London System), indexed by FEN strings. The AI checks this before running Minimax, and chooses randomly from the available moves. This speeds up the early game and makes games more interesting and varied.
      </p>

      <pre><code className="language-python">{`opening_moves = {
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b': [
        (x0_to_move(('c7', 'c5')), 1),
        (x0_to_move(('e7', 'e5')), 1),
    ],
}`}</code></pre>

      <h2>Persistent Memory</h2>
      <p>
        Whenever the engine finds a strong move through deep search, it stores it using Python’s <code>pickle</code> module. Next time that position comes up, the AI can retrieve and reuse it. This speeds up future games and makes the engine more efficient.
      </p>

      <pre><code className="language-python">{`# Save move
with open("best_moves.pickle", "wb") as f:
    pickle.dump(self.previous_finds, f)`}</code></pre>

      <h2>UI: Pygame Integration</h2>
      <p>
        The game runs in a Pygame window with a clean chessboard, pieces loaded from images, and interactive click-based input. The player can play against the AI or another human.
      </p>

      <img src={ui} alt="Chess Game UI" className="page-image" />

      <h2>Future Improvements I may add one day</h2>
      <ul>
        <li>Implement en passant (currently missing)</li>
        <li>Smarter move ordering (e.g., MVV-LVA, killer heuristics)</li>
        <li>Use Zobrist hashing for faster transposition lookups</li>
        <li>Refactor codebase to separate game logic from rendering</li>
        <li>Neural network evaluation instead of hardcoded heuristics</li>
      </ul>

      <h2>Final Thoughts</h2>
      <p>
        This was my first serious programming project, and really solidified my Python skills. Also, it gave me confidence to tackle more complex projects in the future. The engine isn’t perfect, but it plays decent games and was incredibly fun to write.
      </p>
      <p>
      I hope you enjoyed this overview of my chess engine. If you want to check out the code, it’s available on <a href="https://github.com/MarcosSaade/Chess-Engine" target="_blank" rel="noopener noreferrer">GitHub</a>.
      </p>
    </PageTemplate>
  );
}
