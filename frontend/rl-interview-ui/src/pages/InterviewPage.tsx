import React, { useState } from 'react';
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
import { API_ENDPOINTS } from '../config/api';
import Layout from '../components/layout/Layout';

const InterviewPage: React.FC = () => {
  const navigate = useNavigate();
  const { state, setCurrentInterview, setLoading, setError } = useAppContext();
  const [answer, setAnswer] = useState('');
  const [evaluation, setEvaluation] = useState<string | null>(null);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleSubmitAnswer = async () => {
    if (!state.currentInterview?.currentQuestion) return;
    
    setLoading(true);
    try {
      const response = await fetch(API_ENDPOINTS.SUBMIT_ANSWER(state.currentInterview.interviewId), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          answer: answer,
          question_id: state.currentInterview.currentQuestion.questionId
        })
      });

      if (!response.ok) {
        throw new Error('Failed to submit answer');
      }

      const data = await response.json();
      setEvaluation(data.feedback);
      setIsSubmitted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit answer');
    } finally {
      setLoading(false);
    }
  };

  const handleNextQuestion = async () => {
    if (!state.currentInterview) return;
    
    setLoading(true);
    try {
      const response = await fetch(API_ENDPOINTS.NEXT_QUESTION(state.currentInterview.interviewId), {
        method: 'GET'
      });

      if (!response.ok) {
        throw new Error('Failed to get next question');
      }

      const data = await response.json();
      
      if (data.interview_complete) {
        navigate('/results');
        return;
      }

      const updatedInterview = {
        ...state.currentInterview,
        currentQuestion: {
          questionId: data.question.id,
          topic: data.question.topic,
          difficulty: data.question.difficulty,
          question: data.question.content,
          expected_time: data.question.expected_time_minutes,
          follow_up_questions: data.question.follow_up_questions,
          evaluation_points: data.question.evaluation_points,
          subtopic: data.question.subtopic
        },
        currentQuestionIdx: state.currentInterview.currentQuestionIdx + 1,
        questions: [...state.currentInterview.questions, data.question]
      };

      setCurrentInterview(updatedInterview);
      setAnswer('');
      setEvaluation(null);
      setIsSubmitted(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get next question');
    } finally {
      setLoading(false);
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
            {state.currentInterview.currentQuestion.question}
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
            disabled={isSubmitted}
            sx={{ mb: 2 }}
          />
          
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
            {!isSubmitted ? (
              <Button
                variant="contained"
                onClick={handleSubmitAnswer}
                disabled={!answer.trim()}
              >
                Submit Answer
              </Button>
            ) : (
              <Button
                variant="contained"
                onClick={handleNextQuestion}
                color="primary"
              >
                Next Question
              </Button>
            )}
          </Box>
        </Paper>

        {/* Evaluation Section */}
        {evaluation && (
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Evaluation
            </Typography>
            <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
              {evaluation}
            </Typography>
          </Paper>
        )}

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
 