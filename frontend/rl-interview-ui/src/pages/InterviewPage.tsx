import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  TextField,
  Button,
  Paper,
  CircularProgress,
  Alert,
  Stack,
} from '@mui/material';
import QuestionGrid from '../components/QuestionGrid';
import { useInterview } from '../contexts/InterviewContext';
import { API_ENDPOINTS } from '../config/api';

interface Question {
  questionId: string;
  question: string;
  difficulty: number;
}

interface Answer {
  questionId: string;
  answer: string;
  score: number;
  feedback?: {
    correctness: { score: number; comment: string };
    efficiency: { score: number; comment: string };
    code_quality: { score: number; comment: string };
    understanding: { score: number; comment: string };
  };
  improvement_suggestions?: string[];
  overall_feedback?: string;
}

interface Interview {
  interviewId: string;
  topic: string;
  current_question: Question | null;
  questions: Question[];
  answers: Answer[];
  status: 'in_progress' | 'completed';
}

const InterviewPage: React.FC = () => {
  const navigate = useNavigate();
  const {
    currentInterview,
    setCurrentInterview,
    loading,
    error,
    setError,
    setLoading,
  } = useInterview();

  const [selectedQuestionId, setSelectedQuestionId] = useState<string | undefined>();
  const [answer, setAnswer] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [currentAnswer, setCurrentAnswer] = useState<Answer | null>(null);

  useEffect(() => {
    if (!currentInterview?.interviewId && !loading) {
      navigate('/');
    }
  }, [currentInterview, loading, navigate]);

  const handleQuestionSelect = (questionId: string) => {
    setSelectedQuestionId(questionId);
    setAnswer('');
    setCurrentAnswer(null);
  };

  const handleSubmitAnswer = async () => {
    if (!selectedQuestionId || !answer.trim() || !currentInterview?.interviewId) return;

    setSubmitting(true);
    try {
      const response = await fetch(API_ENDPOINTS.SUBMIT_ANSWER(currentInterview.interviewId), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          questionId: selectedQuestionId,
          answer: answer,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to submit answer');
      }

      const data = await response.json();
      setCurrentInterview(data.interview);
      setCurrentAnswer(data.evaluation);
    } catch (err) {
      console.error('Error submitting answer:', err);
      setError('Failed to submit answer. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleEndInterview = () => {
    // Navigate to results page with the current interview data
    navigate('/results');
  };

  const handleNextQuestion = async () => {
    if (!currentInterview?.interviewId) {
      setError('No active interview found');
      return;
    }

    try {
      setLoading(true);
      const response = await fetch(API_ENDPOINTS.NEXT_QUESTION(currentInterview.interviewId), {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to get next question');
      }

      const data = await response.json();
      setCurrentInterview(data);
      setSelectedQuestionId(undefined);
      setAnswer('');
      setCurrentAnswer(null);
    } catch (error) {
      console.error('Error getting next question:', error);
      setError('Failed to get next question. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (!currentInterview) {
    return (
      <Container maxWidth="lg">
        <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" minHeight="80vh">
          <Typography variant="h5" gutterBottom>
            No active interview found
          </Typography>
          <Button variant="contained" color="primary" onClick={() => navigate('/')}>
            Start New Interview
          </Button>
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg">
        <Alert 
          severity="error" 
          sx={{ mt: 3 }}
          action={
            <Button color="inherit" size="small" onClick={() => navigate('/')}>
              Go Back
            </Button>
          }
        >
          {error}
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
          <Typography variant="h4">
            Technical Interview: {currentInterview.topic}
          </Typography>
          <Button
            variant="outlined"
            color="primary"
            onClick={handleEndInterview}
          >
            End Interview & Show Results
          </Button>
        </Box>
        
        {/* Display all questions in a grid */}
        <QuestionGrid
          questions={currentInterview.questions}
          onSelectQuestion={handleQuestionSelect}
          selectedQuestionId={selectedQuestionId}
        />

        {/* Answer section */}
        {selectedQuestionId && (
          <Paper sx={{ mt: 4, p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Your Answer
            </Typography>
            <TextField
              fullWidth
              multiline
              rows={6}
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              placeholder="Type your answer here..."
              variant="outlined"
              sx={{ mb: 2 }}
            />
            <Stack direction="row" spacing={2}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleSubmitAnswer}
                disabled={submitting || !answer.trim()}
              >
                {submitting ? <CircularProgress size={24} /> : 'Submit Answer'}
              </Button>
              {currentAnswer && (
                <Button
                  variant="contained"
                  color="secondary"
                  onClick={handleNextQuestion}
                >
                  Next Question
                </Button>
              )}
            </Stack>
          </Paper>
        )}

        {/* Feedback section */}
        {currentAnswer && (
          <Paper sx={{ mt: 4, p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Feedback
            </Typography>
            <Typography variant="h4" color="primary" gutterBottom>
              Score: {currentAnswer.score}/10
            </Typography>
            
            {currentAnswer.feedback && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Detailed Feedback:
                </Typography>
                {Object.entries(currentAnswer.feedback).map(([criterion, feedback]) => (
                  <Box key={criterion} sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" sx={{ textTransform: 'capitalize' }}>
                      {criterion}: {feedback.score}/10
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {feedback.comment}
                    </Typography>
                  </Box>
                ))}
              </Box>
            )}
            
            {currentAnswer.improvement_suggestions && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Improvement Suggestions:
                </Typography>
                <ul>
                  {currentAnswer.improvement_suggestions.map((suggestion, index) => (
                    <li key={index}>
                      <Typography variant="body2">{suggestion}</Typography>
                    </li>
                  ))}
                </ul>
              </Box>
            )}
            
            {currentAnswer.overall_feedback && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Overall Feedback:
                </Typography>
                <Typography variant="body2">
                  {currentAnswer.overall_feedback}
                </Typography>
              </Box>
            )}
          </Paper>
        )}
      </Box>
    </Container>
  );
};

export default InterviewPage; 
 