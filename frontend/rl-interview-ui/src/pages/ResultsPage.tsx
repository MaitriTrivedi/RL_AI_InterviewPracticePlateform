import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Paper,
  Divider,
} from '@mui/material';
import Grid from '@mui/material/Unstable_Grid2';
import { useInterview } from '../contexts/InterviewContext';

interface Feedback {
  score: number;
  comment: string;
}

interface Evaluation {
  score: number;
  feedback?: Record<string, Feedback>;
  improvement_suggestions?: string[];
}

interface Question {
  id: string;
  text: string;
  evaluation?: Evaluation;
}

interface Interview {
  topic: string;
  questions: Question[];
}

const ResultsPage: React.FC = () => {
  const navigate = useNavigate();
  const { currentInterview } = useInterview() as { currentInterview: Interview | null };

  if (!currentInterview) {
    return (
      <Container maxWidth="lg">
        <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" minHeight="80vh">
          <Typography variant="h5" gutterBottom>
            No interview results found
          </Typography>
          <Button variant="contained" color="primary" onClick={() => navigate('/')}>
            Start New Interview
          </Button>
        </Box>
      </Container>
    );
  }

  const calculateAverageScore = () => {
    const answeredQuestions = currentInterview.questions.filter(q => q.evaluation);
    if (answeredQuestions.length === 0) return 0;
    
    const totalScore = answeredQuestions.reduce((sum, q) => sum + (q.evaluation?.score || 0), 0);
    return totalScore / answeredQuestions.length;
  };

  const averageScore = calculateAverageScore();
  const answeredQuestions = currentInterview.questions.filter(q => q.evaluation);
  const totalQuestions = currentInterview.questions.length;

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <Paper sx={{ p: 4, mb: 4 }}>
          <Typography variant="h4" gutterBottom>
            Interview Results
          </Typography>
          <Typography variant="subtitle1" color="text.secondary" gutterBottom>
            Topic: {currentInterview.topic}
          </Typography>
          
          <Box sx={{ mt: 4, mb: 4 }}>
            <Grid container spacing={4}>
              <Grid xs={12} md={4}>
                <Paper elevation={3} sx={{ p: 3, textAlign: 'center' }}>
                  <Typography variant="h6" gutterBottom>
                    Average Score
                  </Typography>
                  <Typography variant="h3" color="primary">
                    {averageScore.toFixed(1)}/10
                  </Typography>
                </Paper>
              </Grid>
              <Grid xs={12} md={4}>
                <Paper elevation={3} sx={{ p: 3, textAlign: 'center' }}>
                  <Typography variant="h6" gutterBottom>
                    Questions Answered
                  </Typography>
                  <Typography variant="h3" color="primary">
                    {answeredQuestions.length}/{totalQuestions}
                  </Typography>
                </Paper>
              </Grid>
              <Grid xs={12} md={4}>
                <Paper elevation={3} sx={{ p: 3, textAlign: 'center' }}>
                  <Typography variant="h6" gutterBottom>
                    Completion Rate
                  </Typography>
                  <Typography variant="h3" color="primary">
                    {((answeredQuestions.length / totalQuestions) * 100).toFixed(0)}%
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
          </Box>

          <Typography variant="h5" gutterBottom sx={{ mt: 6, mb: 3 }}>
            Detailed Question Analysis
          </Typography>
          
          {currentInterview.questions.map((question, index) => (
            <Paper key={question.id} elevation={2} sx={{ mb: 3, p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Question {index + 1}
              </Typography>
              <Typography variant="body1" gutterBottom>
                {question.text}
              </Typography>
              
              {question.evaluation ? (
                <Box sx={{ mt: 2 }}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle1" gutterBottom>
                    Score: {question.evaluation.score}/10
                  </Typography>
                  
                  {question.evaluation.feedback && (
                    <Grid container spacing={2} sx={{ mt: 1 }}>
                      {Object.entries(question.evaluation.feedback).map(([criterion, feedback]) => (
                        <Grid xs={12} md={6} key={criterion}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="subtitle2" sx={{ textTransform: 'capitalize' }}>
                              {criterion}: {feedback.score}/10
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {feedback.comment}
                            </Typography>
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>
                  )}
                  
                  {question.evaluation.improvement_suggestions && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        Improvement Suggestions:
                      </Typography>
                      <Box component="ul" sx={{ pl: 3 }}>
                        {question.evaluation.improvement_suggestions.map((suggestion, i) => (
                          <Typography component="li" key={i} variant="body2">
                            {suggestion}
                          </Typography>
                        ))}
                      </Box>
                    </Box>
                  )}
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                  Question not attempted
                </Typography>
              )}
            </Paper>
          ))}
        </Paper>

        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={() => navigate('/')}
          >
            Start New Interview
          </Button>
          <Button
            variant="outlined"
            color="primary"
            onClick={() => navigate('/interview')}
          >
            Continue Practice
          </Button>
        </Box>
      </Box>
    </Container>
  );
};

export default ResultsPage; 