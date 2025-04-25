import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        mean = self.network(state)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim=2, lr=3e-4, gamma=0.99, epsilon=0.2):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Expanded topic categories and their weights
        self.topic_weights = {
            'data_structures': 0.0,
            'algorithms': 0.0,
            'system_design': 0.0,
            'programming_concepts': 0.0,
            'machine_learning': 0.0,
            'deep_learning': 0.0,
            'operating_systems': 0.0,
            'databases': 0.0,
            'computer_networks': 0.0,
            'software_engineering': 0.0,
            'distributed_systems': 0.0,
            'security': 0.0
        }
    
    def get_state_representation(self, interview_state):
        """Convert interview state to neural network input"""
        if not interview_state['answers']:
            return np.array([
                0.5,  # Initial normalized score
                0.0,  # Questions answered
                0.3,  # Starting difficulty (easy)
                0.0,  # Performance trend
                0.0,  # Time trend
                *list(self.topic_weights.values())  # Topic weights
            ])
        
        # Calculate features
        scores = [ans['score'] for ans in interview_state['answers']]
        avg_score = np.mean(scores) / 10.0
        num_questions = len(scores) / 10.0  # Normalize by max questions
        current_difficulty = interview_state.get('current_difficulty', 3) / 10.0
        
        # Performance trend (positive or negative)
        if len(scores) >= 2:
            trend = (scores[-1] - scores[-2]) / 10.0
        else:
            trend = 0.0
        
        # Time trend (if available)
        time_trend = 0.0  # TODO: Implement time tracking
        
        # Update topic weights based on performance
        if interview_state['answers']:
            last_answer = interview_state['answers'][-1]
            last_topic = last_answer.get('topic', 'programming_concepts')
            performance = last_answer['score'] / 10.0
            
            # Increase weight for topics where performance is low
            for topic in self.topic_weights:
                if topic == last_topic:
                    self.topic_weights[topic] = (self.topic_weights[topic] + (1 - performance)) / 2
                else:
                    self.topic_weights[topic] *= 0.9  # Decay other topics
        
        state = np.array([
            avg_score,
            num_questions,
            current_difficulty,
            trend,
            time_trend,
            *list(self.topic_weights.values())
        ])
        
        return state
    
    def get_next_question_params(self, interview_state):
        """Determine next question's difficulty and topic focus"""
        state = self.get_state_representation(interview_state)
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            dist = self.policy(state_tensor)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)  # Bound the actions
        
        # First action dimension: difficulty (1-10)
        difficulty = int(((action[0].item() + 1) / 2) * 9 + 1)  # Map [-1,1] to [1,10]
        
        # Second action dimension: topic selection
        topic_probs = torch.softmax(action[1:] * 5, dim=0)  # Temperature scaling
        topics = list(self.topic_weights.keys())
        selected_topic = np.random.choice(topics, p=topic_probs.numpy())
        
        # Generate detailed prompt based on performance
        if interview_state['answers']:
            last_answer = interview_state['answers'][-1]
            if last_answer['score'] < 5:  # Poor performance
                focus = "fundamental concepts"
                depth = "basic"
            elif last_answer['score'] < 8:  # Medium performance
                focus = "practical applications"
                depth = "intermediate"
            else:  # Good performance
                focus = "advanced concepts"
                depth = "challenging"
        else:
            focus = "fundamental concepts"
            depth = "basic"
        
        return {
            'difficulty': difficulty,
            'topic': selected_topic,
            'focus': focus,
            'depth': depth
        }
    
    def update(self, state, action, reward, next_state, done):
        """Update policy using PPO"""
        # Convert to tensors
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])
        
        # Get value predictions
        value = self.value(state)
        next_value = self.value(next_state)
        
        # Compute advantage
        td_target = reward + self.gamma * next_value * (1 - done)
        advantage = td_target - value
        
        # Get action probabilities
        dist = self.policy(state)
        log_prob = dist.log_prob(action).sum()
        entropy = dist.entropy().mean()
        
        # Store old log probability
        old_log_prob = log_prob.detach()
        
        # PPO update
        for _ in range(5):  # Multiple epochs
            # Get new probabilities
            dist = self.policy(state)
            new_log_prob = dist.log_prob(action).sum()
            
            # Compute ratio and surrogate loss
            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_pred = self.value(state)
            value_loss = nn.MSELoss()(value_pred, td_target.detach())
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()
    
    def select_next_question(self, interview_state, available_questions):
        """Select the next question using the PPO policy"""
        state = self.get_state_representation(interview_state)
        action, _ = self.get_action(state)
        
        # Convert action to question selection
        # Action is a continuous value that we'll use to rank questions
        question_scores = []
        for q in available_questions:
            # Score based on how close question difficulty is to desired difficulty
            difficulty_score = -abs(q['difficulty'] - (action[0] * 10))
            question_scores.append((q, difficulty_score))
        
        # Select question with highest score
        selected_question = max(question_scores, key=lambda x: x[1])[0]
        return selected_question

    def get_state(self):
        # Basic metrics
        avg_score = np.mean(self.score_history) if self.score_history else 0.0
        num_questions = len(self.score_history)
        current_difficulty = self.current_difficulty
        
        # Time and performance trends
        time_trend = self.calculate_time_trend()
        performance_trend = self.calculate_performance_trend()
        
        # Topic weights
        topic_weights = list(self.topic_weights.values())
        
        # Combine all components into state vector
        state = np.array([
            avg_score,
            num_questions,
            current_difficulty,
            time_trend,
            performance_trend,
            *topic_weights
        ])
        
        return state

    def calculate_time_trend(self):
        if len(self.time_history) < 2:
            return 0.0
        # Calculate the trend in response times (positive means taking longer)
        return np.mean(np.diff(self.time_history))

    def calculate_performance_trend(self):
        if len(self.score_history) < 2:
            return 0.0
        # Calculate the trend in scores (positive means improving)
        return np.mean(np.diff(self.score_history)) 