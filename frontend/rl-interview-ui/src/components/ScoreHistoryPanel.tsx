import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Stack,
  Chip,
  Divider,
  LinearProgress,
  Tooltip
} from '@mui/material';
import { styled } from '@mui/material/styles';

interface QuestionScore {
  questionNumber: number;
  difficulty: number;
  score: number;
  topic: string;
  timeTaken: number;
}

interface ScoreHistoryPanelProps {
  scores: QuestionScore[];
}

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.default,
  height: '100%',
  maxHeight: 'calc(100vh - 200px)',
  overflowY: 'auto'
}));

const getDifficultyInfo = (difficulty: number) => {
  if (difficulty <= 3) return { label: 'Easy', color: 'success' as const, description: 'Basic concepts' };
  if (difficulty <= 7) return { label: 'Medium', color: 'warning' as const, description: 'Advanced concepts' };
  return { label: 'Hard', color: 'error' as const, description: 'Expert level' };
};

const getScoreColor = (score: number): 'success' | 'warning' | 'error' => {
  if (score >= 0.7) return 'success';
  if (score >= 0.4) return 'warning';
  return 'error';
};

export const ScoreHistoryPanel: React.FC<ScoreHistoryPanelProps> = ({ scores }) => {
  const calculateAverageScore = () => {
    if (scores.length === 0) return 0;
    return scores.reduce((acc, curr) => acc + curr.score, 0) / scores.length;
  };

  const calculateAverageDifficulty = () => {
    if (scores.length === 0) return 0;
    return scores.reduce((acc, curr) => acc + curr.difficulty, 0) / scores.length;
  };

  return (
    <StyledPaper elevation={2}>
      <Typography variant="h6" gutterBottom>
        Score History
      </Typography>
      
      {/* Average Score and Difficulty */}
      <Box mb={2}>
        <Stack direction="row" spacing={2}>
          <Box flex={1}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Average Score
            </Typography>
            <Chip
              label={`${(calculateAverageScore() * 10).toFixed(1)}/10`}
              color={getScoreColor(calculateAverageScore())}
              sx={{ width: '100%' }}
            />
          </Box>
          <Box flex={1}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Average Difficulty
            </Typography>
            <Tooltip title={getDifficultyInfo(calculateAverageDifficulty()).description}>
              <Chip
                label={`${calculateAverageDifficulty().toFixed(1)}/10`}
                color={getDifficultyInfo(calculateAverageDifficulty()).color}
                sx={{ width: '100%' }}
              />
            </Tooltip>
          </Box>
        </Stack>
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Question Scores */}
      <Stack spacing={2}>
        {scores.map((score, index) => (
          <Box key={index}>
            <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1}>
              <Typography variant="subtitle2">
                Question {score.questionNumber}
              </Typography>
              <Chip
                size="small"
                label={`${(score.score * 10).toFixed(1)}/10`}
                color={getScoreColor(score.score)}
              />
            </Stack>
            
            <Stack direction="row" spacing={1} mb={1}>
              <Chip
                size="small"
                label={score.topic.toUpperCase()}
                variant="outlined"
                color="primary"
              />
              <Tooltip title={`${getDifficultyInfo(score.difficulty).description} - Level ${score.difficulty.toFixed(1)}/10`}>
                <Chip
                  size="small"
                  label={`${getDifficultyInfo(score.difficulty).label} (${score.difficulty.toFixed(1)})`}
                  color={getDifficultyInfo(score.difficulty).color}
                />
              </Tooltip>
            </Stack>

            <Box sx={{ position: 'relative', width: '100%' }}>
              <Tooltip title={`Time taken: ${Math.round(score.timeTaken)} seconds`}>
                <div style={{ width: '100%', cursor: 'pointer' }}>
                  <LinearProgress
                    variant="determinate"
                    value={score.score * 100}
                    color={getScoreColor(score.score)}
                    sx={{ height: 6, borderRadius: 3 }}
                  />
                </div>
              </Tooltip>
              <Typography 
                variant="caption" 
                sx={{ 
                  position: 'absolute',
                  right: 0,
                  top: -16,
                  fontSize: '0.7rem',
                  color: 'text.secondary'
                }}
              >
                {Math.round(score.timeTaken)}s
              </Typography>
            </Box>
          </Box>
        ))}
      </Stack>

      {scores.length === 0 && (
        <Typography variant="body2" color="text.secondary" textAlign="center">
          No questions answered yet
        </Typography>
      )}
    </StyledPaper>
  );
}; 