import React, { useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Stack,
  CircularProgress,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../contexts/AppContext';
import Layout from '../components/layout/Layout';
import { API_ENDPOINTS } from '../config/api';

interface FinalStats {
  questions_asked: number;
  average_performance: number;
  final_difficulty: number;
  topics_covered: string[];
}

const InterviewCompletionPage: React.FC = () => {
  const navigate = useNavigate();
  const { state, setError } = useAppContext();
  const [finalStats, setFinalStats] = React.useState<FinalStats | null>(null);
  const [loading, setLoading] = React.useState(true);

  useEffect(() => {
    const endInterview = async () => {
      try {
        if (!state.currentInterview) {
          navigate('/');
          return;
        }

        const response = await fetch(
          API_ENDPOINTS.END_INTERVIEW(state.currentInterview.interviewId),
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              user_id: state.currentInterview.userId
            })
          }
        );

        if (!response.ok) {
          throw new Error('Failed to end interview');
        }

        const data = await response.json();
        setFinalStats(data.final_stats);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to end interview');
      } finally {
        setLoading(false);
      }
    };

    endInterview();
  }, [navigate, state.currentInterview, setError]);

  if (loading) {
    return (
      <Layout>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
          <CircularProgress />
        </Box>
      </Layout>
    );
  }

  const getPerformanceLevel = (score: number) => {
    if (score >= 8) return { text: 'Excellent', color: 'success.main' };
    if (score >= 6) return { text: 'Good', color: 'primary.main' };
    if (score >= 4) return { text: 'Fair', color: 'warning.main' };
    return { text: 'Needs Improvement', color: 'error.main' };
  };

  const performance = getPerformanceLevel(finalStats?.average_performance || 0);

  return (
    <Layout>
      <Typography variant="h4" component="h1" gutterBottom>
        Interview Complete!
      </Typography>

      <Paper sx={{ p: 4, mt: 3 }}>
        <Stack spacing={3}>
          <Box>
            <Typography variant="h6" gutterBottom>
              Final Results
            </Typography>
            <Typography variant="body1" color={performance.color} fontWeight="bold" gutterBottom>
              Overall Performance: {performance.text}
            </Typography>
            <Typography variant="body1">
              Average Score: {(finalStats?.average_performance || 0).toFixed(1)}/10
            </Typography>
            <Typography variant="body1">
              Questions Completed: {finalStats?.questions_asked || 0}
            </Typography>
            <Typography variant="body1">
              Final Difficulty Level: {(finalStats?.final_difficulty || 0).toFixed(1)}/10
            </Typography>
          </Box>

          <Box>
            <Typography variant="h6" gutterBottom>
              Key Takeaways
            </Typography>
            <Typography variant="body1" paragraph>
              • Your strongest areas were in the topics you consistently scored well
            </Typography>
            <Typography variant="body1" paragraph>
              • The system adapted to your skill level, reaching a final difficulty of {(finalStats?.final_difficulty || 0).toFixed(1)}
            </Typography>
            <Typography variant="body1">
              • You completed {finalStats?.questions_asked || 0} technical questions across different topics
            </Typography>
            {finalStats?.topics_covered && (
              <Typography variant="body1" sx={{ mt: 1 }}>
                • Topics covered: {finalStats.topics_covered.join(', ')}
              </Typography>
            )}
          </Box>
        </Stack>
      </Paper>

      <Box sx={{ mt: 4, display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
        <Button variant="outlined" onClick={() => navigate('/')}>
          Return Home
        </Button>
        <Button
          variant="contained"
          color="primary"
          onClick={() => navigate('/interview/setup')}
        >
          Start New Interview
        </Button>
      </Box>
    </Layout>
  );
};

export default InterviewCompletionPage; 