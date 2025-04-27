import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Button,
  Grid,
  Card,
  CardContent,
  Divider,
  Chip,
  Stack
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { useAppContext } from '../contexts/AppContext';
import Layout from '../components/layout/Layout';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(3)
}));

const ResultCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: theme.palette.grey[50]
}));

const ResultsPage: React.FC = (): JSX.Element => {
  const navigate = useNavigate();
  const { state: { currentInterview } } = useAppContext();

  if (!currentInterview) {
    return (
      <Layout>
        <Typography variant="h4" gutterBottom>
          No Interview Results Available
        </Typography>
        <Button variant="contained" onClick={() => navigate('/')}>
          Return to Home
        </Button>
      </Layout>
    );
  }

  const averageScore = currentInterview.stats.average_performance * 10;
  const totalQuestions = currentInterview.stats.questions_asked;

  return (
    <Layout>
      <StyledPaper>
        <Typography variant="h4" gutterBottom>
          Interview Results
        </Typography>

        <Stack direction="row" spacing={2} mb={3}>
          <Chip
            label={`Topic: ${currentInterview.topic}`}
            color="primary"
            variant="outlined"
          />
          <Chip
            label={`Questions: ${totalQuestions}`}
            color="secondary"
            variant="outlined"
          />
          <Chip
            label={`Average Score: ${averageScore.toFixed(1)}/10`}
            color="info"
            variant="outlined"
          />
        </Stack>

        <Grid container spacing={3}>
          {/* Performance Summary */}
          <Grid item xs={12} md={6}>
            <ResultCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Performance Summary
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Questions Answered: {totalQuestions}
                  </Typography>
                  <Typography variant="subtitle1" gutterBottom>
                    Average Score: {averageScore.toFixed(1)}/10
                  </Typography>
                  <Typography variant="subtitle1" gutterBottom>
                    Final Difficulty Level: {currentInterview.difficulty.toFixed(1)}/10
                  </Typography>
                </Box>
              </CardContent>
            </ResultCard>
          </Grid>

          {/* Recommendations */}
          <Grid item xs={12} md={6}>
            <ResultCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recommendations
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Typography variant="body1" paragraph>
                  {averageScore >= 8
                    ? "Excellent performance! You're well-prepared for technical interviews. Consider tackling more challenging questions to further enhance your skills."
                    : averageScore >= 6
                    ? "Good performance! Focus on strengthening your understanding of core concepts and practice more complex problem-solving scenarios."
                    : "Keep practicing! Focus on understanding fundamental concepts and work through problems step by step. Consider reviewing basic data structures and algorithms."}
                </Typography>
              </CardContent>
            </ResultCard>
          </Grid>
        </Grid>

        <Box sx={{ mt: 4, display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            onClick={() => navigate('/')}
          >
            Return to Home
          </Button>
          <Button
            variant="contained"
            onClick={() => navigate('/interview/setup')}
          >
            Start New Interview
          </Button>
        </Box>
      </StyledPaper>
    </Layout>
  );
};

export default ResultsPage; 