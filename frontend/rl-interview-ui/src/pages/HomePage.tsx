import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  Button,
  Paper, 
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Container,
  CircularProgress,
  Alert
} from '@mui/material';
import Grid from '@mui/material/Unstable_Grid2';
import { Experimental_CssVarsProvider as CssVarsProvider } from '@mui/material/styles';
import { useNavigate } from 'react-router-dom';
import { 
  Description as DescriptionIcon,
  QuestionAnswer as QuestionAnswerIcon,
  Psychology as PsychologyIcon,
  Assessment as AssessmentIcon,
  School as SchoolIcon
} from '@mui/icons-material';
import Layout from '../components/layout/Layout';
import { useInterview } from '../contexts/InterviewContext';
import { API_ENDPOINTS } from '../config/api';

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const { setUserId } = useInterview();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStartPractice = () => {
    navigate('/interview/setup');
  };

  return (
    <Layout>
      <Box>
        {/* Hero Section */}
        <Paper
          sx={{
            p: 4,
            mb: 4,
            backgroundImage: 'linear-gradient(120deg, #1976d2 0%, #4791db 100%)',
            color: 'white',
            borderRadius: 2,
          }}
        >
          <Typography variant="h3" component="h1" gutterBottom fontWeight="bold">
            AI Interview Practice Platform
          </Typography>
          <Typography variant="h6" sx={{ mb: 3 }}>
            Improve your interview skills with our AI-powered practice platform
          </Typography>
          <Typography variant="body1" paragraph sx={{ maxWidth: '80%', mb: 4 }}>
            Our system uses Reinforcement Learning to adapt to your performance and provide 
            personalized interview questions based on your resume and past answers.
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              color="secondary"
              size="large"
              onClick={() => navigate('/resume-upload')}
              sx={{ fontWeight: 'bold' }}
            >
              Upload Your Resume
            </Button>
            <Button
              variant="outlined"
              color="inherit"
              size="large"
              onClick={handleStartPractice}
              sx={{ bgcolor: 'rgba(255,255,255,0.1)', fontWeight: 'bold' }}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <CircularProgress size={24} sx={{ mr: 1 }} />
                  Starting...
                </>
              ) : (
                'Start Practice'
              )}
            </Button>
          </Box>
        </Paper>

        {/* Features Cards */}
        <Typography variant="h4" component="h2" gutterBottom sx={{ mt: 6, mb: 3 }}>
          Key Features
        </Typography>
        <Box sx={{ flexGrow: 1 }}>
          <Grid container spacing={3}>
            <Grid xs={12} md={4}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <DescriptionIcon color="primary" sx={{ fontSize: 32, mr: 1 }} />
                    <Typography variant="h6" component="h3">
                      Resume Analysis
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ flexGrow: 1 }}>
                    Upload your resume and our system will analyze your skills, education, and work 
                    experience to generate relevant interview questions.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid xs={12} md={4}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <PsychologyIcon color="primary" sx={{ fontSize: 32, mr: 1 }} />
                    <Typography variant="h6" component="h3">
                      Reinforcement Learning
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ flexGrow: 1 }}>
                    Our AI adapts to your performance and selects questions that challenge you appropriately,
                    creating a personalized learning experience.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid xs={12} md={4}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <AssessmentIcon color="primary" sx={{ fontSize: 32, mr: 1 }} />
                    <Typography variant="h6" component="h3">
                      Performance Tracking
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ flexGrow: 1 }}>
                    Get detailed feedback on your answers and track your progress over time
                    with comprehensive analytics and scoring.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>

        {/* How It Works Section */}
        <Typography variant="h4" component="h2" gutterBottom sx={{ mt: 6, mb: 3 }}>
          How It Works
        </Typography>
        <Paper sx={{ p: 3, mb: 4 }}>
          <List>
            <ListItem alignItems="flex-start">
              <ListItemIcon>
                <DescriptionIcon color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Step 1: Upload Your Resume"
                secondary="Upload your resume in PDF format. Our system will parse and extract key information about your skills, education, and work experience."
              />
            </ListItem>
            <Divider component="li" />
            <ListItem alignItems="flex-start">
              <ListItemIcon>
                <SchoolIcon color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Step 2: Select Topic & Difficulty"
                secondary="Choose a topic for your interview or let our system suggest one based on your resume. You can also set an initial difficulty level."
              />
            </ListItem>
            <Divider component="li" />
            <ListItem alignItems="flex-start">
              <ListItemIcon>
                <QuestionAnswerIcon color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Step 3: Answer Questions"
                secondary="Answer the questions provided by our AI. Each question is selected based on your resume and adapted to your skill level."
              />
            </ListItem>
            <Divider component="li" />
            <ListItem alignItems="flex-start">
              <ListItemIcon>
                <AssessmentIcon color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Step 4: Get Feedback & Improve"
                secondary="Receive immediate feedback and scoring on your answers. Our system will adapt future questions based on your performance."
              />
            </ListItem>
          </List>
        </Paper>

        {/* Get Started CTA */}
        <Box
          sx={{
            mt: 6,
            p: 4,
            textAlign: 'center',
            bgcolor: 'primary.light',
            borderRadius: 2,
            color: 'white',
          }}
        >
          <Typography variant="h5" gutterBottom>
            Ready to improve your interview skills?
          </Typography>
          <Button
            variant="contained"
            color="secondary"
            size="large"
            onClick={() => navigate('/resume-upload')}
            sx={{ mt: 2, fontWeight: 'bold' }}
          >
            Get Started Now
          </Button>
        </Box>
      </Box>
    </Layout>
  );
};

export default HomePage; 