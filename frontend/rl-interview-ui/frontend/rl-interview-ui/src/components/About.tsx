import React from 'react';
import { Container, Typography, Paper, Box, Grid } from '@mui/material';
import CodeIcon from '@mui/icons-material/Code';
import PsychologyIcon from '@mui/icons-material/Psychology';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';

export const About = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 8, mb: 4 }}>
        <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
          <Typography variant="h3" component="h1" gutterBottom align="center">
            About AI Interview Practice Platform
          </Typography>
          
          <Typography variant="body1" paragraph sx={{ mb: 4 }}>
            Our AI-powered interview practice platform is designed to help you prepare for technical interviews
            through personalized practice sessions, real-time feedback, and adaptive question selection.
          </Typography>

          <Grid container spacing={4} sx={{ mt: 2 }}>
            <Grid component="div" item xs={12} md={4}>
              <Box sx={{ textAlign: 'center', p: 2 }}>
                <CodeIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
                <Typography variant="h5" gutterBottom>
                  Technical Excellence
                </Typography>
                <Typography variant="body1">
                  Practice with a wide range of technical questions across different difficulty levels.
                </Typography>
              </Box>
            </Grid>

            <Grid component="div" item xs={12} md={4}>
              <Box sx={{ textAlign: 'center', p: 2 }}>
                <PsychologyIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
                <Typography variant="h5" gutterBottom>
                  AI-Powered Feedback
                </Typography>
                <Typography variant="body1">
                  Receive instant, detailed feedback on your answers from our advanced AI system.
                </Typography>
              </Box>
            </Grid>

            <Grid component="div" item xs={12} md={4}>
              <Box sx={{ textAlign: 'center', p: 2 }}>
                <AutoGraphIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
                <Typography variant="h5" gutterBottom>
                  Adaptive Learning
                </Typography>
                <Typography variant="body1">
                  Questions adapt to your skill level for optimal learning progression.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Box>
    </Container>
  );
}; 