import React from 'react';
import PageTemplate from '../components/PageTemplate';
import { BlockMath, InlineMath } from 'react-katex';

import sugarzeroHeroImage from '../images/sugarzero.png';
import sugarzeroGameScreenshot from '../images/sugarzero/game-interface.png';
import sugarzeroBoardStates from '../images/sugarzero/game-state.png';
import sugarzeroTrainingLoss from '../images/sugarzero/loss_curve.png';

export default function SugarZeroPage() {
  return (
    <PageTemplate title="SugarZero: Teaching AI to Master a Simple Game through Self-Play" image={sugarzeroHeroImage}>
      <p>
        AlphaZero fascinated me from the moment I first read about it. The idea that an AI could master complex games like Chess and Go starting with nothing but the rules, playing against itself to achieve superhuman performance, struck me as almost magical. After writing the post about my high school Chess Minimax project, I decided it was time to truly understand how AlphaZero works by building a simplified version myself.
      </p>
      
      <p>
        Rather than tackling Chess (which would require computational resources I didn't have), I chose a simpler game called <em>Sugar</em> – a game I had also previously coded a minimax AI for. This gave me the perfect test bed: complex enough to be interesting but simple enough to be tractable on my hardware. What started as a learning exercise turned into a fascinating journey that revealed unexpected insights about reinforcement learning and game design.
      </p>

      <p>
        Halfway through the project, which I had been tackling using resources from multiple sources, I finally read the original AlphaZero paper—and I actually understood it! This was immensely satisfying and helped me refine my implementation with a deeper appreciation of the underlying concepts.
      </p>

      <h2>The Game: Sugar</h2>
      <p>
        <em>Sugar</em> is an abstract strategy game played on a 3×3 board. The rules are simple:
      </p>
      <ul>
        <li>Players control 6 stackable pieces, initially placed uniformly on their back rank.</li>
        <li>On each turn, a player moves one piece from the top of a stack to an adjacent square (horizontally or vertically).</li>
        <li>A piece can only move to a square if the destination stack is shorter than or equal in height to the stack it's moving from.</li>
        <li>For the first three moves, each player must make at least one "forward" move (toward the opponent's side).</li>
        <li>The game ends when a player cannot make any legal moves, and that player loses.</li>
      </ul>

      <img src={sugarzeroGameScreenshot} alt="SugarZero Game Interface" className="page-image" />

      <h2>AlphaZero in a Nutshell</h2>
      <p>
        Before diving into implementation details, let's review the core AlphaZero approach:
      </p>
      <ol>
        <li><strong>Neural Networks:</strong>
          <ul>
            <li>A policy network that outputs probabilities for each possible move</li>
            <li>A value network that evaluates how good a position is</li>
          </ul>
        </li>
        <li><strong>Monte Carlo Tree Search (MCTS):</strong> Uses the networks as guidance while searching through possible move sequences.</li>
        <li><strong>Self-Play:</strong> The AI plays against itself, generating training data.</li>
        <li><strong>Learning Loop:</strong> The system continuously improves through iteration.</li>
      </ol>

      <h2>Building SugarZero</h2>

      <h3>Game Representation and Logic</h3>
      <p>
        The foundation of any game AI is an efficient game state representation. For Sugar, I created a <code>GameState</code> class that:
      </p>
      <ul>
        <li>Represents the board as a NumPy array for efficient operations</li>
        <li>Implements rules and move validation</li>
        <li>Provides a fast <code>.clone()</code> method (used in MCTS)</li>
        <li>Tracks game state (move count, current player, game over)</li>
      </ul>

      <p>
        Here's how the game state captures both piece positions and stack heights:
      </p>

      <pre><code className="language-python">{`class GameState:
    def __init__(self):
        # 3x3 board with stacks containing (color, height) info
        self.board = np.zeros((3, 3, 2), dtype=np.int8)
        # Initialize starting positions (1=white, -1=black)
        self.board[0, :, 0] = -1  # Black pieces on top row
        self.board[0, :, 1] = 1   # Height of 1 for each piece
        self.board[2, :, 0] = 1   # White pieces on bottom row
        self.board[2, :, 1] = 1   # Height of 1 for each piece
        self.turn = 1  # White starts
        self.move_count = 0
        self.forward_moves_white = 0
        self.forward_moves_black = 0
        self.game_over = False
        self.winner = None

    def clone(self):
        # Fast clone without deepcopy
        gs = GameState.__new__(GameState)
        gs.board = self.board.copy()
        gs.turn = self.turn
        gs.move_count = self.move_count
        gs.forward_moves_white = self.forward_moves_white
        gs.forward_moves_black = self.forward_moves_black
        gs.game_over = self.game_over
        gs.winner = self.winner
        return gs`}</code></pre>

      <img src={sugarzeroBoardStates} alt="Sugar Board States" className="page-image" />

      <h3>Monte Carlo Tree Search with PUCT</h3>
      <p>
        For the search algorithm, I implemented MCTS with the PUCT formula (Predictor + Upper Confidence bounds for Trees) used by AlphaZero. PUCT is fundamentally a solution to the exploration-exploitation dilemma in reinforcement learning, commonly known as the multi-armed bandit problem.
      </p>
      
      <p>
        The multi-armed bandit problem asks: if you have multiple slot machines ("bandits") with unknown payout rates, how do you balance trying new machines (exploration) versus playing machines you already know give good rewards (exploitation)? PUCT solves this elegantly by combining:
      </p>
      
      <ul>
      <li>
        <strong>Exploitation:</strong> Using <InlineMath math="Q(s,a)" /> — the average value of an action based on past experience
      </li>
      <li>
        <strong>Exploration:</strong> Favoring actions that are both promising (high neural network prior <InlineMath math="P(s,a)" />) and underexplored (low visit count <InlineMath math="N(s,a)" />)
      </li>
    </ul>
      
      <p>
        The PUCT formula is:
      </p>
      
      <BlockMath math={'\\text{PUCT}(s,a) = Q(s,a) + c_{\\text{puct}} \\cdot P(s,a) \\cdot \\frac{\\sqrt{N(s)}}{1 + N(s,a)}'} />

      <p>Where:</p>
      <ul>
        <li><InlineMath math={'Q(s,a)'} />: Average value of taking action <InlineMath math={'a'} /> from state <InlineMath math={'s'} /></li>
        <li><InlineMath math={'P(s,a)'} />: Prior probability from the policy network</li>
        <li><InlineMath math={'N(s)'} />: Total visits to state <InlineMath math={'s'} /></li>
        <li><InlineMath math={'N(s,a)'} />: Visits to action <InlineMath math={'a'} /> from state <InlineMath math={'s'} /></li>
        <li><InlineMath math={'c_{\\text{puct}}'} />: Exploration constant (higher values encourage more exploration)</li>
      </ul>

      <pre><code className="language-python">{`class MCTSNode:
    def __init__(self, game_state, parent=None, prior=0.0):
        self.game_state = game_state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.expanded = False
    
    def select_child(self, c_puct=1.4):
        # PUCT formula: Q(s,a) + c_puct * P(s,a) * √(N(s)) / (1 + N(s,a))
        best_score = -float('inf')
        best_action = None
        
        for action, child in self.children.items():
            # Exploitation term
            q_value = 0.0 if child.visit_count == 0 else child.value_sum / child.visit_count
            
            # Exploration term
            u_value = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            
            # Combined score
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action, self.children[best_action]`}</code></pre>

      <p>
        The above logic selects child nodes during MCTS traversal based on both exploitation (value) and exploration (visit count and prior probability).
      </p>

      <h3>Neural Network Architecture</h3>
      <p>
        After experimenting with different architectures, I settled on a convolutional neural network for both policy and value predictions. The key insight was that spatially structured data (like a board game) benefits from convolutional layers that can capture patterns across the board.
      </p>

      <pre><code className="language-python">{`class PolicyValueNet(nn.Module):
    def __init__(self):
        super(PolicyValueNet, self).__init__()
        # Input: 2 players x 3 rows x 3 cols x 6 stack heights
        # Reshaped to: (batch, 12, 3, 3)
        
        # Shared convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(64 * 3 * 3 + 1, 128),  # +1 for turn information
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 9*9)  # 9 source squares, 9 destination squares
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(64 * 3 * 3 + 1, 64),  # +1 for turn information
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, board_tensor, turn):
        # board_tensor: (batch, 2, 3, 3, 6) -> (batch, 12, 3, 3)
        bsz = board_tensor.size(0)
        x = board_tensor.permute(0, 1, 4, 2, 3).reshape(bsz, 12, 3, 3)
        
        # Shared features
        features = self.conv(x)
        flat_features = features.view(bsz, -1)
        
        # Add turn information
        turn_tensor = turn.view(bsz, 1).float()
        full_features = torch.cat([flat_features, turn_tensor], dim=1)
        
        # Policy output (move probabilities)
        policy_logits = self.policy_head(full_features)
        policy = F.softmax(policy_logits, dim=1)
        
        # Value output (position evaluation)
        value = self.value_head(full_features)
        
        return policy, value.squeeze(-1)`}</code></pre>

      <h3>Self-Play Training Pipeline</h3>
      <p>
        The training loop was the most complex part, involving several components:
      </p>

      <ol>
        <li><strong>Game Generation:</strong> Multiple self-play games running in parallel</li>
        <li><strong>Replay Buffer:</strong> Store (state, policy, value) tuples from past games</li>
        <li><strong>Network Training:</strong> Using MSE loss for value predictions and kl-divergence for policy predictions, and weight decay</li>
        <li><strong>Evaluation:</strong> Testing new models against old checkpoints</li>
      </ol>

      <div className="info-box">
        <h4>Parallel Processing for Self-Play</h4>
        <p>
          A critical optimization I implemented was parallel game generation using Python's threading and <code>concurrent.futures.ProcessPoolExecutor</code>. This allowed multiple self-play episodes to run simultaneously across CPU cores, dramatically speeding up the training pipeline. Without this parallelization, generating thousands of self-play games would have been prohibitively time-consuming.
        </p>
      </div>

      <p>
        Before starting the main self-play loop, I generated 400 warmup games using pure UCT (without neural networks). This provided a solid foundation of diverse gameplay patterns to bootstrap the neural network training, avoiding the need to learn from completely random play in the earliest stages.
      </p>

      <pre><code className="language-python">{`def train_networks(replay_buffer, policy_value_net, optimizer, batch_size=128):
    if len(replay_buffer) < batch_size:
        return 0, 0  # Not enough data yet
    
    # Sample batch from replay buffer
    states, turns, target_policies, target_values = replay_buffer.sample(batch_size)
    
    # Convert to tensors
    state_tensor = torch.FloatTensor(states)
    turn_tensor = torch.FloatTensor(turns)
    policy_tensor = torch.FloatTensor(target_policies)
    value_tensor = torch.FloatTensor(target_values)
    
    # Forward pass
    optimizer.zero_grad()
    policy_pred, value_pred = policy_value_net(state_tensor, turn_tensor)
    
    # Calculate loss
    policy_loss = -torch.sum(policy_tensor * torch.log(policy_pred + 1e-8)) / batch_size
    value_loss = F.mse_loss(value_pred, value_tensor)
    total_loss = policy_loss + value_loss
    
    # Backward pass and optimize
    total_loss.backward()
    optimizer.step()
    
    return policy_loss.item(), value_loss.item()`}</code></pre>

      <h3>Training Enhancements for Better Performance</h3>
      <p>
        To enhance training quality and overcome challenges specific to the Sugar game, I implemented several key optimizations:
      </p>

      <h4>Dirichlet Noise</h4>
      <p>
        Dirichlet noise is a statistical distribution used to add randomness to the MCTS root node's prior probabilities. In SugarZero, I applied this noise using:
      </p>
      <p className="formula">
        <InlineMath math={"\\pi' = (1 - \\varepsilon) \\cdot \\pi + \\varepsilon \\cdot \\eta \\quad \\text{where} \\; \\eta \\sim \\text{Dir}(\\alpha)"} />
      </p>
      <p>
        With <InlineMath math={"\\alpha = 0.8"} /> and <InlineMath math={"\\varepsilon = 0.25"} />, this injects controlled randomness only at the root node. The motivation is to prevent premature convergence to deterministic opening sequences, forcing the model to explore alternative strategies even after it forms initial preferences.
      </p>


      <h4>Temperature Sampling</h4>
      <p>
        Temperature controls how "focused" or "exploratory" move selection is during play. I implemented temperature annealing:
      </p>
      <ul>
        <li>Starting at T=1.0 (high diversity in move selection)</li>
        <li>Gradually decreasing to T=0.5 (more focused on stronger moves)</li>
        <li>Applied only to the first 10 moves of each game</li>
      </ul>
      <p>
        I introduced temperature sampling because early versions of the AI weren't playing decisively enough. By applying temperature only to early moves, the agent explores diverse openings but plays deterministically in critical positions where there's often only one good move.
      </p>

      <h4>Additional Training Optimizations</h4>
      <ul>
        <li>
          <strong>Draw Filtering:</strong> Games exceeding 200 moves without a winner were discarded. Since Sugar has no draw condition, these extensive games represent cycles or degenerate patterns rather than strategic play. Including them would dilute the training signal with low-quality data.
        </li>
        <li>
          <strong>Winner Oversampling:</strong> Decisive wins were duplicated in the replay buffer, effectively doubling their representation. This enhanced learning from successful strategies, which were relatively rare in early training.
        </li>
        <li>
          <strong>Replay Buffer Balancing:</strong> The buffer contained a mix of 70% neural self-play games, 20% UCT-only games, and 10% random play. This diversity prevented premature convergence to suboptimal strategies.
        </li>
        <li>
          <strong>Batch Normalization:</strong> Added after each convolutional layer to stabilize training and speed convergence. This made training deeper networks viable without gradient explosion/vanishing issues.
        </li>
        <li>
          <strong>Dropout (p=0.2):</strong> Applied to fully connected layers in both policy and value heads. This regularization technique improved generalization by preventing overfitting to specific game patterns.
        </li>
        <li>
          <strong>Replay Buffer Size:</strong> Set to 15,000 transitions to maintain a broad window of recent and successful games. Since each game generates many training examples, this size balanced memory constraints with training diversity.
        </li>
      </ul>

      <h2>Results and Discoveries</h2>

      <h3>Training Progress</h3>
      <p>
        After training for 10,000 self-play games, the system showed interesting behavior:
      </p>

      <img src={sugarzeroTrainingLoss} alt="Training Loss Graph" className="page-image" />

      <p>
        The loss curves suggested that the model was learning, but it reached a plateau relatively early in training. Evaluations against previous checkpoints showed mixed results:
      </p>

      <p>
        The model achieved a win rate of around 48% against older versions of itself (the rest being draws), indicating it had learned some basic strategies but struggled to consistently improve.
      </p>

      <p> Additionally, the model's performance was subjectively weak when playing agaisnt myself, and even after 10,000 games it was still somewhat easy to beat.</p>



      <h3>The Cycle Problem</h3>
      <p>
        The most interesting finding wasn't about the implementation but about the nature of the game itself. Sugar lacks the "simplification pressure" found in games like Chess and Go. In Chess, pieces get captured and removed from the board, naturally driving the game toward an end state. In Sugar, the full complement of pieces remains throughout the game.
      </p>

      <p>
        This creates a fundamental issue for reinforcement learning: the agent can learn to "stall" rather than win. If a sequence of moves can be repeated indefinitely without penalty, the AI has no incentive to take risks to achieve a definitive win when it can simply avoid losing.
      </p>

      <p>
        Despite my efforts to counter this by filtering drawn-out games and oversampling decisive victories, the AI seemed to hit a ceiling. It learned basic strategic concepts but couldn't consistently improve beyond them.
      </p>

      <p>
        I decided to leave the project at this point, as I had achieved my goal of understanding AlphaZero's mechanisms. However, I believe that with further refinements to the game design or the learning process, it could be possible to overcome this cycle problem and achieve better performance.
      </p>


      <h2>Conclusion</h2>
      <p>
        SugarZero successfully implemented the core mechanisms of AlphaZero: neural policy and value networks, PUCT search, and self-play learning. The model learned basic strategic behaviors and could play reasonably well, but it reached a plateau at around a 48% win rate against older versions of itself.
      </p>

      <p>
        The most important lesson wasn't technical but conceptual: the effectiveness of reinforcement learning through self-play is heavily influenced by the structural properties of the game being learned. Games without clear progression mechanisms can lead to agents that learn to avoid losing rather than actively pursue winning, and thus we have to either design more frequent rewards in the environment or fundamentally change the agent's learning process to avoid this issue.
      </p>

      <p>
        This suggests that the remarkable success of systems like AlphaZero stems not just from their sophisticated algorithms but also from their synergy with games that have inherent directional pressure toward conclusion.
      </p>

      <h2>Try it Out!</h2>
      <p>
        The code for SugarZero is available on <a href="https://github.com/MarcosSaade/SugarZero/" target="_blank" rel="noopener noreferrer">GitHub</a>. (see if you can beat it!)
      </p>

      <p>
        The game loads the 10,000-game checkpoint by default, letting you play against the trained AI or watch AI-vs-AI matches.
      </p>
      
      <p>
        <strong>Controls:</strong>
        <ul>
          <li><strong>R:</strong> Restart game</li>
          <li><strong>Q:</strong> Quit</li>
          <li><strong>A:</strong> Toggle AI on/off</li>
          <li><strong>S:</strong> Swap which side the AI plays</li>
          <li><strong>D:</strong> Cycle AI difficulty (Easy → Medium → Hard)</li>
        </ul>
      </p>

      <p>
        Whether you're interested in game AI, reinforcement learning, or just enjoy abstract strategy games, I hope you find this project as fascinating as I did!
      </p>
    </PageTemplate>
  );
}