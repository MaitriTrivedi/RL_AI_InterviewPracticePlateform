import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import json
import random
from pathlib import Path

class InterviewNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(state), self.critic(state)

class InterviewAgent:
    def __init__(
        self,
        questions_file: str = "interview_questions.json",
        state_dim: int = None,  # Will be calculated based on number of topics
        action_dim: int = 10,  # Difficulty levels from 1 to 10
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Load questions dataset using absolute path
        questions_path = Path(__file__).parent / questions_file
        with open(questions_path, 'r') as f:
            self.questions = json.load(f)  # The file is already a list of questions
        
        # Calculate state dimension based on number of topics
        # 5 base features + topic weights
        self.state_dim = 5 + len(self.get_all_topics())
        self.action_dim = action_dim
        
        # Initialize networks
        self.network = InterviewNetwork(self.state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize interview state
        self.reset_state()
    
    def reset_state(self):
        """Reset the interview state for a new candidate."""
        self.current_state = {
            'avg_score': 0.0,
            'num_questions': 0,
            'current_difficulty': 5,  # Start with medium difficulty
            'time_trend': 0.0,
            'performance_trend': 0.0,
            'topic_weights': {topic: 1.0 for topic in self.get_all_topics()}
        }
        self.previous_scores = []
        self.previous_times = []
    
    def get_all_topics(self) -> List[str]:
        """Get list of all available topics."""
        return list(set(q['topic'].lower() for q in self.questions))  # Convert topics to lowercase
    
    def state_to_tensor(self, state: Dict) -> torch.Tensor:
        """Convert state dictionary to tensor."""
        state_list = [
            state['avg_score'],
            state['num_questions'],
            state['current_difficulty'],
            state['time_trend'],
            state['performance_trend']
        ]
        # Ensure topic weights are added in a consistent order
        topics = sorted(self.get_all_topics())
        state_list.extend([state['topic_weights'].get(topic, 1.0) for topic in topics])
        return torch.FloatTensor(state_list).unsqueeze(0).to(self.device)  # Add batch dimension
    
    def update_state(self, score: float, time_taken: float, topic: str):
        """Update interview state based on candidate's performance."""
        # Update scores and calculate trends
        self.previous_scores.append(score)
        self.previous_times.append(time_taken)
        
        # Update average score
        self.current_state['avg_score'] = np.mean(self.previous_scores)
        self.current_state['num_questions'] += 1
        
        # Calculate trends
        if len(self.previous_scores) >= 2:
            self.current_state['performance_trend'] = self.previous_scores[-1] - self.previous_scores[-2]
            self.current_state['time_trend'] = self.previous_times[-1] - self.previous_times[-2]
        
        # Update topic weights based on performance
        topic = topic.lower()  # Convert topic to lowercase
        topic_weight = self.current_state['topic_weights'].get(topic, 1.0)
        if score < 5:  # Poor performance
            topic_weight *= 0.8  # Reduce weight to ask more questions from this topic
        elif score > 7:  # Good performance
            topic_weight *= 1.2  # Increase weight to move to other topics
        self.current_state['topic_weights'][topic] = max(0.1, min(2.0, topic_weight))
    
    def select_action(self, state: Dict) -> Tuple[int, float]:
        """Select next difficulty level using the policy network."""
        state_tensor = self.state_to_tensor(state)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
        
        # Add exploration noise
        if random.random() < self.epsilon:
            difficulty = random.randint(1, 10)
        else:
            difficulty = torch.argmax(action_probs).item() + 1
        
        return difficulty, value.item()
    
    def select_next_question(self, previous_score: float = None, time_taken: float = None, previous_topic: str = None) -> Dict:
        """Select the next interview question based on current state."""
        # Update state if we have previous question information
        if previous_score is not None and previous_topic is not None:
            self.update_state(previous_score, time_taken or 300, previous_topic)
        
        # Select next difficulty level
        next_difficulty, _ = self.select_action(self.current_state)
        
        # Filter questions by difficulty
        difficulty_range = (next_difficulty - 1, next_difficulty + 1)
        candidate_questions = [
            q for q in self.questions
            if difficulty_range[0] <= q['difficulty'] <= difficulty_range[1]
        ]
        
        # Weight questions by topic weights
        weights = [
            self.current_state['topic_weights'].get(q['topic'].lower(), 1.0)  # Convert topic to lowercase
            for q in candidate_questions
        ]
        
        # Select question
        if candidate_questions:
            question = random.choices(candidate_questions, weights=weights, k=1)[0]
            return question
        else:
            # Fallback to random question if no candidates found
            return random.choice(self.questions)
    
    def train_step(self, state, action, reward, next_state, done):
        """Update the policy network using PPO."""
        # Convert to tensors
        state = self.state_to_tensor(state)
        next_state = self.state_to_tensor(next_state)
        action_tensor = torch.LongTensor([action-1]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)
        
        # Get old action probabilities and value
        old_action_probs, old_value = self.network(state)
        old_action_prob = old_action_probs[0, action-1].item()
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            action_probs, value = self.network(state)
            _, next_value = self.network(next_state)
            
            # Calculate advantage
            if done:
                advantage = reward_tensor - value
            else:
                advantage = reward_tensor + self.gamma * next_value - value
            
            # Calculate action probability ratio
            action_prob = action_probs[0, action-1]
            ratio = action_prob / old_action_prob
            
            # Calculate PPO loss
            loss1 = ratio * advantage
            loss2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
            actor_loss = -torch.min(loss1, loss2)
            critic_loss = advantage.pow(2)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def save(self, path: str):
        """Save the model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_state': self.current_state,
        }, path)
    
    def load(self, path: str):
        """Load the model."""
        if Path(path).exists():
            checkpoint = torch.load(path)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_state = checkpoint['current_state'] 