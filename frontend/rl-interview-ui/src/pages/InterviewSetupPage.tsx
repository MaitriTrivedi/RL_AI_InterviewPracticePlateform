import React from 'react';
import { 
  Box, 
  Typography, 
  Button,
  Paper,
  Stack,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../contexts/AppContext';
import { API_ENDPOINTS } from '../config/api';
import Layout from '../components/layout/Layout';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { Interview, Question } from '../types/index';

const InterviewSetupPage: React.FC = () => {
  const navigate = useNavigate();
  const { state, setCurrentInterview, setLoading, setError } = useAppContext();

  const handleStartInterview = async () => {
    setLoading(true);
    try {
      const userId = Date.now().toString(); // Generate a temporary user ID
      const response = await fetch(API_ENDPOINTS.CREATE_INTERVIEW, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: userId,
          topic: "data_structures", // Default topic
          difficulty: 5, // Medium difficulty
          maxQuestions: 10 // Default max questions
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to create interview');
      }

      const data = await response.json();
      
      // Transform the first question to match our Question interface
      const firstQuestion: Question = {
        questionId: data.first_question.id,
        topic: data.first_question.topic,
        difficulty: data.first_question.difficulty,
        question: data.first_question.content,  // backend sends as 'content'
        follow_up_questions: data.first_question.follow_up_questions,
        evaluation_points: data.first_question.evaluation_points,
        subtopic: data.first_question.subtopic
      };

      // Create interview object matching the Interview interface
      const newInterview: Interview = {
        interviewId: data.session_id || userId, // Use session_id from backend if available, fallback to userId
        userId: userId, // Store the user ID separately
        topic: data.first_question.topic,  // Use the topic from the first question
        currentQuestion: firstQuestion,
        currentQuestionIdx: 0,
        maxQuestions: data.session_stats.max_questions || 10, // Use max_questions from session
        questions: [firstQuestion],
        answers: [],
        status: 'in_progress',
        difficulty: data.initial_difficulty,
        stats: data.session_stats
      };
      
      setCurrentInterview(newInterview);
      navigate('/interview');
    } catch (err) {
      console.error('Error creating interview:', err);
      setError(err instanceof Error ? err.message : 'Failed to create interview');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Layout>
      <Typography variant="h4" component="h1" gutterBottom>
        Interview Setup
      </Typography>

      <Box sx={{ display: 'flex', gap: 3, mt: 3 }}>
        {/* RL Agent Info */}
        <Box sx={{ flex: 1 }}>
          {/* RL Agent Configuration */}
          <Paper sx={{ p: 3, bgcolor: 'primary.main', color: 'white', mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              RL Agent Configuration
            </Typography>
            <Typography variant="body2" paragraph>
              Our AI system uses Reinforcement Learning to adapt the difficulty and 
              relevance of questions based on your performance.
            </Typography>
            <Typography variant="body2" fontWeight="bold">
              Use Reinforcement Learning Agent: Enabled
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.8 }}>
              The system will learn from your answers and adapt to your skill level
            </Typography>
          </Paper>

          {/* Resume Information */}
          <Paper variant="outlined" sx={{ p: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Resume Information
            </Typography>
            <Stack spacing={2}>
              <Box>
                <Typography variant="subtitle2">Education:</Typography>
                <Typography variant="body2">0 entries</Typography>
              </Box>
              <Box>
                <Typography variant="subtitle2">Work Experience:</Typography>
                <Typography variant="body2">0 entries</Typography>
              </Box>
              <Box>
                <Typography variant="subtitle2">Projects:</Typography>
                <Typography variant="body2">0 entries</Typography>
              </Box>
            </Stack>
          </Paper>
        </Box>
      </Box>

      {/* Action Buttons */}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2, mt: 4 }}>
        <Button variant="outlined" onClick={() => navigate('/')}>
          Cancel
        </Button>
        <Button
          variant="contained"
          color="primary"
          onClick={handleStartInterview}
        >
          Start Interview
        </Button>
      </Box>

      {/* Loading Overlay */}
      {state.loading && <LoadingSpinner message="Setting up your interview..." />}
    </Layout>
  );
};

export default InterviewSetupPage; 