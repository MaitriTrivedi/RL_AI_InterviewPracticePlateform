# Interview Configuration
INTERVIEW_CONFIG = {
    'MAX_QUESTIONS': 5,  # Default number of questions per interview
    'DEFAULT_DIFFICULTY': 5.0,  # Default starting difficulty (medium)
    'DEFAULT_TOPIC': 'ds',  # Default starting topic
    'PERFORMANCE_THRESHOLDS': {
        'GOOD': 0.7,  # Above this is considered good performance
        'POOR': 0.4   # Below this is considered poor performance
    },
    'DIFFICULTY_RANGES': {
        'EASY': (1, 3),
        'MEDIUM': (4, 7),
        'HARD': (8, 10)
    }
} 