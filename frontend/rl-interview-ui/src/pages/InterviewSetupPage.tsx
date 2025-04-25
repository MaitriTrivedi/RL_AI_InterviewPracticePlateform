import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Slider,
  FormHelperText,
  Paper,
  Stack,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../context/AppContext';
import Layout from '../components/layout/Layout';
import LoadingSpinner from '../components/common/LoadingSpinner';

const difficultyMarks = [
  { value: 1, label: 'Easy' },
  { value: 5, label: 'Medium' },
  { value: 10, label: 'Hard' },
];

const InterviewSetupPage: React.FC = () => {
  const navigate = useNavigate();
  const { state } = useAppContext();
  
  const [topic, setTopic] = useState('');
  const [difficulty, setDifficulty] = useState<number>(5);
  const [maxQuestions, setMaxQuestions] = useState<number>(10);
  
  // Handle topic change
  const handleTopicChange = (event: SelectChangeEvent) => {
    setTopic(event.target.value as string);
  };
  
  // Handle difficulty change
  const handleDifficultyChange = (_event: Event, newValue: number | number[]) => {
    setDifficulty(newValue as number);
  };
  
  // Handle max questions change
  const handleMaxQuestionsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(event.target.value);
    if (!isNaN(value) && value >= 1 && value <= 20) {
      setMaxQuestions(value);
    }
  };
  
  return (
    <Layout>
      <Typography variant="h4" component="h1" gutterBottom>
        Interview Setup
      </Typography>

      <Box sx={{ display: 'flex', gap: 3, mt: 3 }}>
        {/* Left Column - Interview Configuration */}
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6" gutterBottom>
            Configure Your Interview
          </Typography>

          {/* Topic Selection */}
          <FormControl fullWidth sx={{ mb: 4 }}>
            <InputLabel id="topic-select-label">Interview Topic</InputLabel>
            <Select
              labelId="topic-select-label"
              value={topic}
              onChange={handleTopicChange}
              label="Interview Topic"
            >
              <MenuItem value="Data Structures">Data Structures</MenuItem>
              <MenuItem value="Algorithms">Algorithms</MenuItem>
              <MenuItem value="System Design">System Design</MenuItem>
              <MenuItem value="Web Development">Web Development</MenuItem>
              <MenuItem value="Machine Learning">Machine Learning</MenuItem>
            </Select>
            <FormHelperText>
              Topics are suggested based on your resume
            </FormHelperText>
          </FormControl>

          {/* Difficulty Level */}
          <Box sx={{ mb: 4 }}>
            <Typography gutterBottom>
              Difficulty Level: {difficulty}
            </Typography>
            <Slider
              value={difficulty}
              onChange={handleDifficultyChange}
              step={1}
              marks={difficultyMarks}
              min={1}
              max={10}
              valueLabelDisplay="auto"
            />
            <FormHelperText>
              {difficulty <= 3 && 'Beginner-friendly questions'}
              {difficulty > 3 && difficulty <= 7 && 'Moderate difficulty, suitable for most candidates'}
              {difficulty > 7 && 'Advanced questions for experienced candidates'}
            </FormHelperText>
          </Box>

          {/* Number of Questions */}
          <TextField
            fullWidth
            label="Number of Questions"
            type="number"
            value={maxQuestions}
            onChange={handleMaxQuestionsChange}
            InputProps={{ inputProps: { min: 1, max: 20 } }}
            sx={{ mb: 4 }}
            helperText="Choose between 1 and 20 questions"
          />
        </Box>

        {/* Right Column - RL Agent Info */}
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
          onClick={() => navigate('/interview/new')}
          disabled={!topic}
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