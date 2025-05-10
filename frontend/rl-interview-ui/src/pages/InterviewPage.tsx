import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  CircularProgress,
  Chip,
  Stack,
  Divider,
  Tooltip,
  LinearProgress,
  Grid
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../contexts/AppContext';
import { rlAgentApi, scoreHistoryApi } from '../services/api';
import Layout from '../components/layout/Layout';
import { Interview, Question, SubmitAnswerResponse } from '../types';
import { ScoreHistoryPanel } from '../components/ScoreHistoryPanel';
import axios from 'axios';

// Add difficulty level helper
const getDifficultyInfo = (difficulty: number): { 
  label: string; 
  color: 'success' | 'warning' | 'error' | 'default' | 'primary' | 'secondary' | 'info' 
} => {
  if (difficulty <= 3) {
    return { label: 'Easy', color: 'success' };
  } else if (difficulty <= 7) {
    return { label: 'Medium', color: 'warning' };
  } else {
    return { label: 'Hard', color: 'error' };
  }
};

// Add difficulty progress color helper
const getDifficultyProgressColor = (difficulty: number): 'success' | 'warning' | 'error' | 'primary' | 'secondary' | 'info' => {
  if (difficulty <= 3) return 'success';
  if (difficulty <= 7) return 'warning';
  return 'error';
};

const InterviewPage: React.FC = () => {
  const navigate = useNavigate();
  const { state, setCurrentInterview, setLoading, setError } = useAppContext();
  const [answer, setAnswer] = useState('');
  const [isSubmitting, setSubmitting] = useState(false);
  const [startTime] = useState(Date.now());
  const [scoreHistory, setScoreHistory] = useState<Array<{
    questionNumber: number;
    difficulty: number;
    score: number;
    topic: string;
    timeTaken: number;
  }>>([]);

  useEffect(() => {
    // Check for active interview
    const interviewId = localStorage.getItem('interviewId');
    if (!state.currentInterview && !interviewId) {
      navigate('/');
    }
  }, [state.currentInterview, navigate]);

  const handleSubmitAnswer = async () => {
    if (!state.currentInterview) return;
    
    setSubmitting(true);
    try {
      const timeTaken = (Date.now() - startTime) / 1000; // Convert to seconds
      
      const response = await rlAgentApi.submitAnswer(
        state.currentInterview.interviewId,
        answer,
        timeTaken
      );

      const data = response.data as SubmitAnswerResponse;
      
      // Save score history
      const newScore = {
        questionNumber: state.currentInterview.currentQuestionIdx + 1,
        difficulty: state.currentInterview.currentQuestion.difficulty,
        score: data.evaluation.score / 10,
        topic: state.currentInterview.currentQuestion.topic || 'unknown',
        timeTaken
      };
      
      setScoreHistory(prev => [...prev, newScore]);
      
      try {
        await scoreHistoryApi.saveScore(state.currentInterview.interviewId, newScore);
      } catch (scoreErr) {
        console.error('Error saving score history:', scoreErr);
      }
      
      // Check if interview is complete
      if (data.next_state.interview_complete) {
        localStorage.removeItem('interviewId'); // Clear session
        navigate('/interview/complete');
        return;
      }
      
      // Update interview state with next question
      if (state.currentInterview && data.next_state.next_question) {
        setCurrentInterview({
          ...state.currentInterview,
          currentQuestion: data.next_state.next_question,
          currentQuestionIdx: state.currentInterview.currentQuestionIdx + 1,
          difficulty: data.next_state.next_question.difficulty
        });
        setAnswer('');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit answer');
    } finally {
      setSubmitting(false);
    }
  };

  if (!state.currentInterview) {
    return (
      <Layout>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
          <CircularProgress />
        </Box>
      </Layout>
    );
  }

  return (
    <Layout>
      <Grid container spacing={3}>
        {/* Main Interview Area */}
        <Grid item xs={12} md={8}>
          <Box sx={{ maxWidth: '100%', mx: 'auto', p: 3 }}>
            {/* Question Section */}
            <Paper sx={{ p: 3, mb: 3 }}>
              <Stack direction="row" spacing={2} mb={2} alignItems="center">
                <Typography variant="h6">
                  Question {state.currentInterview.currentQuestionIdx + 1}
                </Typography>
                <Tooltip title="Question difficulty level" arrow>
                  <Box sx={{ minWidth: 200 }}>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Chip 
                        label={`Difficulty: ${state.currentInterview.currentQuestion.difficulty.toFixed(1)}/10`} 
                        color={getDifficultyInfo(state.currentInterview.currentQuestion.difficulty).color}
                        variant="filled"
                      />
                      <Chip 
                        label={getDifficultyInfo(state.currentInterview.currentQuestion.difficulty).label}
                        color={getDifficultyInfo(state.currentInterview.currentQuestion.difficulty).color}
                        variant="outlined"
                      />
                    </Stack>
                    <LinearProgress
                      variant="determinate"
                      value={state.currentInterview.currentQuestion.difficulty * 10}
                      color={getDifficultyProgressColor(state.currentInterview.currentQuestion.difficulty)}
                      sx={{ 
                        mt: 1, 
                        height: 8, 
                        borderRadius: 4,
                        bgcolor: 'grey.200'
                      }}
                    />
                  </Box>
                </Tooltip>
              </Stack>
              
              <Divider sx={{ mb: 3 }} />
              
              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', mb: 2 }}>
                {state.currentInterview.currentQuestion.content}
              </Typography>
              
              <TextField
                fullWidth
                multiline
                rows={6}
                variant="outlined"
                placeholder="Type your answer here..."
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                disabled={isSubmitting}
                sx={{ mb: 2 }}
              />
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 2 }}>
                <Button
                  variant="contained"
                  color="error"
                  onClick={() => {
                    localStorage.removeItem('interviewId');
                    navigate('/interview/complete');
                  }}
                >
                  End Interview
                </Button>

                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleSubmitAnswer}
                  disabled={isSubmitting || !answer.trim()}
                >
                  {isSubmitting ? <CircularProgress size={24} /> : 'Submit Answer'}
                </Button>
              </Box>
            </Paper>
          </Box>
        </Grid>

        {/* Score History Panel */}
        <Grid item xs={12} md={4}>
          <ScoreHistoryPanel scores={scoreHistory} />
        </Grid>
      </Grid>
    </Layout>
  );
};

export default InterviewPage; 
 