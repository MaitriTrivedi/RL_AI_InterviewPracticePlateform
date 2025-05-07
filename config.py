"""Configuration settings for the AI Interview Practice Platform."""

# Interview Configuration
INTERVIEW_CONFIG = {
    # Model settings
    'model': {
        'version': 'model_v1_20250507_182838_reward_0.629',  # Current best model
        'state_dim': 9,
        'action_dim': 1,
        'hidden_dim': 64,
    },
    
    # Interview settings
    'interview': {
        'max_questions': 5,
        'min_difficulty': 1,
        'max_difficulty': 10,
        'performance_threshold': 0.6,
        'high_performance_threshold': 0.8,
    },
    
    # Evaluation settings
    'evaluation': {
        'technical_weight': 0.4,
        'communication_weight': 0.2,
        'problem_solving_weight': 0.3,
        'time_efficiency_weight': 0.1,
    },
    
    # Time settings
    'timing': {
        'base_time_per_difficulty': 2,  # minutes
        'max_time_multiplier': 2.5,
        'evaluation_display_time': 3,  # seconds
    }
} 