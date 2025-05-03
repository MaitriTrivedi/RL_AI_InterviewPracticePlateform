import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  CircularProgress,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../contexts/AppContext';
import { rlAgentApi } from '../services/api';
import Layout from '../components/layout/Layout';
import { Interview, Question, SubmitAnswerResponse } from '../types';
import axios from 'axios';

const InterviewPage: React.FC = () => {
  const navigate = useNavigate();
  const { state, setCurrentInterview, setLoading, setError } = useAppContext();
  const [answer, setAnswer] = useState('');
  const [isSubmitting, setSubmitting] = useState(false);
  const [startTime] = useState(Date.now());

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
      
      // Check if interview is complete
      if (data.next_state.interview_complete) {
        localStorage.removeItem('interviewId'); // Clear session
        navigate('/interview/complete');
        return;
      }

      // Update interview state with next question
      if (state.currentInterview) {
        const updatedInterview: Interview = {
          ...state.currentInterview,
          currentQuestion: data.next_state.next_question,
          questions: [...state.currentInterview.questions, data.next_state.next_question],
          currentQuestionIdx: state.currentInterview.currentQuestionIdx + 1,
          stats: data.current_state.session_stats
        };
        setCurrentInterview(updatedInterview);
        setAnswer('');
      }
      
    } catch (err) {
      console.error('Error submitting answer:', err);
      if (axios.isAxiosError(err) && err.response?.status === 404) {
        localStorage.removeItem('interviewId'); // Clear invalid session
        navigate('/');
      } else {
        setError(err instanceof Error ? err.message : 'Failed to submit answer');
      }
    } finally {
      setSubmitting(false);
    }
  };

  if (!state.currentInterview?.currentQuestion) {
    return (
      <Layout>
        <Typography>No active interview found. Please start a new interview.</Typography>
        <Button variant="contained" onClick={() => navigate('/')}>
          Go to Home
        </Button>
      </Layout>
    );
  }

  return (
    <Layout>
      <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
        {/* Question Section */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Question {state.currentInterview.currentQuestionIdx + 1} of {state.currentInterview.maxQuestions}
          </Typography>
          <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', mb: 2 }}>
            {state.currentInterview.currentQuestion.content}
          </Typography>
          
          {/* Follow-up Questions */}
          {state.currentInterview.currentQuestion.follow_up_questions && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Follow-up Questions:
              </Typography>
              <ul>
                {state.currentInterview.currentQuestion.follow_up_questions.map((q, idx) => (
                  <li key={idx}>
                    <Typography variant="body2">{q}</Typography>
                  </li>
                ))}
              </ul>
            </Box>
          )}

          {/* Evaluation Points */}
          {state.currentInterview.currentQuestion.evaluation_points && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Key Points to Address:
              </Typography>
              <ul>
                {state.currentInterview.currentQuestion.evaluation_points.map((point, idx) => (
                  <li key={idx}>
                    <Typography variant="body2">{point}</Typography>
                  </li>
                ))}
              </ul>
            </Box>
          )}
        </Paper>

        {/* Answer Section */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <TextField
            fullWidth
            multiline
            rows={6}
            variant="outlined"
            label="Your Answer"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            disabled={isSubmitting}
            sx={{ mb: 2 }}
          />
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              Topic: {state.currentInterview.currentQuestion.topic} - {state.currentInterview.currentQuestion.subtopic}
            </Typography>
            <Button
              variant="contained"
              onClick={handleSubmitAnswer}
              disabled={!answer.trim() || isSubmitting}
            >
              {isSubmitting ? 'Submitting...' : 'Submit Answer'}
            </Button>
          </Box>
        </Paper>

        {/* Loading Overlay */}
        {state.loading && (
          <Box
            sx={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'rgba(0, 0, 0, 0.5)',
              zIndex: 9999,
            }}
          >
            <CircularProgress />
          </Box>
        )}
      </Box>
    </Layout>
  );
};

export default InterviewPage; 
 