import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Container, Box } from '@mui/material';
import HomePage from './pages/HomePage';
import InterviewPage from './pages/InterviewPage';
import ResultsPage from './pages/ResultsPage';
import ResumeUploadPage from './pages/ResumeUploadPage';
import InterviewSetupPage from './pages/InterviewSetupPage';
import InterviewCompletionPage from './pages/InterviewCompletionPage';
import { AppProvider } from './contexts/AppContext';

// Create a theme instance
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#9c27b0',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppProvider>
        <Router>
          <Box
            sx={{
              minHeight: '100vh',
              backgroundColor: 'background.default',
              py: 4,
            }}
          >
            <Container maxWidth="lg">
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/interview/setup" element={<InterviewSetupPage />} />
                <Route path="/interview" element={<InterviewPage />} />
                <Route path="/interview/complete" element={<InterviewCompletionPage />} />
                <Route path="/interview/results" element={<ResultsPage />} />
                <Route path="/resume-upload" element={<ResumeUploadPage />} />
              </Routes>
            </Container>
          </Box>
        </Router>
      </AppProvider>
    </ThemeProvider>
  );
}

export default App;
