import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Stack,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../contexts/AppContext';
import Layout from '../components/layout/Layout';
import { Interview } from '../types';

const getPerformanceColor = (score: number): 'success' | 'primary' | 'warning' | 'error' => {
  if (score >= 8) return 'success';
  if (score >= 6) return 'primary';
  if (score >= 4) return 'warning';
  return 'error';
};

const getPerformanceText = (score: number): string => {
  if (score >= 8) return 'Excellent';
  if (score >= 6) return 'Good';
  if (score >= 4) return 'Fair';
  return 'Needs Improvement';
};

const ResultsPage: React.FC = () => {
  const navigate = useNavigate();
  const { state } = useAppContext();
  const currentInterview = state.currentInterview;

  if (!currentInterview) {
    return (
      <Layout>
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="h6" gutterBottom>
            No interview results available
          </Typography>
          <Button
            variant="contained"
            onClick={() => navigate('/interview/setup')}
            sx={{ mt: 2 }}
          >
            Start New Interview
          </Button>
        </Box>
      </Layout>
    );
  }

  const averagePerformance = currentInterview.stats.average_performance;
  const performanceColor = getPerformanceColor(averagePerformance);
  const performanceText = getPerformanceText(averagePerformance);

  return (
    <Layout>
      <Typography variant="h4" component="h1" gutterBottom>
        Interview Results
      </Typography>

      {/* Overall Performance */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Stack spacing={2}>
          <Typography variant="h6">Overall Performance</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip
              label={performanceText}
              color={performanceColor}
              sx={{ fontWeight: 'bold' }}
            />
            <Typography>
              Average Score: {averagePerformance.toFixed(1)}/10
            </Typography>
          </Box>
          <Typography>
            Questions Completed: {currentInterview.questions.length} of {currentInterview.maxQuestions}
          </Typography>
          <Typography>
            Final Difficulty: {currentInterview.difficulty.toFixed(1)}/10
          </Typography>
        </Stack>
      </Paper>

      {/* Question History */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Question History
        </Typography>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>#</TableCell>
                <TableCell>Topic</TableCell>
                <TableCell>Subtopic</TableCell>
                <TableCell>Difficulty</TableCell>
                <TableCell>Performance</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {currentInterview.questions.map((question, index) => (
                <TableRow key={question.questionId}>
                  <TableCell>{index + 1}</TableCell>
                  <TableCell>{question.topic}</TableCell>
                  <TableCell>{question.subtopic}</TableCell>
                  <TableCell>{question.difficulty.toFixed(1)}</TableCell>
                  <TableCell>
                    {currentInterview.answers[index] && (
                      <Chip
                        size="small"
                        label={`${(Number(currentInterview.answers[index]) * 10).toFixed(1)}/10`}
                        color={getPerformanceColor(Number(currentInterview.answers[index]) * 10)}
                      />
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Action Buttons */}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2, mt: 3 }}>
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

export default ResultsPage; 