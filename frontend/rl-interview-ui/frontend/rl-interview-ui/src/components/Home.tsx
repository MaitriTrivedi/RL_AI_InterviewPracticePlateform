import React from 'react';
import { Box, Typography, Button, Container, Paper } from '@mui/material';
import { useNavigate } from 'react-router-dom';

export const Home = () => {
  const navigate = useNavigate();

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 8, mb: 4 }}>
        <Paper elevation={3} sx={{ p: 4, textAlign: 'center', borderRadius: 2 }}>
          <Typography variant="h2" component="h1" gutterBottom>
            AI Interview Practice Platform
          </Typography>
          <Typography variant="h5" color="text.secondary" paragraph>
            Enhance your interview skills with our AI-powered practice platform.
            Get real-time feedback and improve your performance.
          </Typography>
          <Box sx={{ mt: 4 }}>
            <Button 
              variant="contained" 
              color="primary" 
              size="large"
              onClick={() => navigate('/practice')}
              sx={{ mr: 2 }}
            >
              Start Practice
            </Button>
            <Button 
              variant="outlined" 
              color="primary" 
              size="large"
              onClick={() => navigate('/about')}
            >
              Learn More
            </Button>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
}; 