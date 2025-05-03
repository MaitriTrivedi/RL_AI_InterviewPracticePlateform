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
  const [hasEnded, setHasEnded] = React.useState(false);

  useEffect(() => {
    const endInterview = async () => {
      // If interview has already been ended or no current interview, don't proceed
      if (hasEnded || !state.currentInterview) {
        if (!state.currentInterview) {
          navigate('/');
        }
        return;
      }

      try {
        const response = await fetch(
          API_ENDPOINTS.END_INTERVIEW(state.currentInterview.interviewId),
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              session_id: state.currentInterview.interviewId
            })
          }
        );

        const data = await response.json();
        
        // Transform the stats to match our interface
        const transformedStats: FinalStats = {
          questions_asked: data.final_stats.questions_asked,
          average_performance: data.final_stats.average_performance * 10, // Convert to 0-10 scale
          final_difficulty: data.final_stats.final_difficulty,
          topics_covered: state.currentInterview.questions
            .map(q => q.topic)
            .filter((value, index, self) => self.indexOf(value) === index) // Get unique topics
        };
        setFinalStats(transformedStats);
        setHasEnded(true);
      } catch (err) {
        console.error('Error ending interview:', err);
        // If there's an error, use the current interview state to show results
        if (state.currentInterview) {
          const fallbackStats: FinalStats = {
            questions_asked: state.currentInterview.questions.length,
            average_performance: (state.currentInterview.stats.average_performance || 0) * 10,
            final_difficulty: state.currentInterview.difficulty,
            topics_covered: state.currentInterview.questions
              .map(q => q.topic)
              .filter((value, index, self) => self.indexOf(value) === index)
          };
          setFinalStats(fallbackStats);
          setHasEnded(true);
        }
      } finally {
        setLoading(false);
      }
    };

    endInterview();
  }, [navigate, state.currentInterview, hasEnded]); // Remove finalStats from dependencies

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