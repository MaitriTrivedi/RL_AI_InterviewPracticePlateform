import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Chip,
  Stack,
  Divider
} from '@mui/material';
import { styled } from '@mui/material/styles';
import ReactMarkdown from 'react-markdown';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(3),
  backgroundColor: theme.palette.background.default
}));

const AnswerBox = styled(TextField)(({ theme }) => ({
  marginTop: theme.spacing(2),
  marginBottom: theme.spacing(2),
  '& .MuiInputBase-root': {
    fontFamily: 'monospace'
  }
}));

const ResultCard = styled(Card)(({ theme }) => ({
  marginTop: theme.spacing(2),
  backgroundColor: theme.palette.grey[50],
  border: `1px solid ${theme.palette.divider}`
}));

interface InterviewInterfaceProps {
  onEnd: () => void;
}

interface Question {
  id: string;
  topic: string;
  difficulty: number;
  content: string;
}

interface EvaluationResult {
  score: number;
  feedback: string;
  correctAnswer?: string;
}

export const InterviewInterface: React.FC<InterviewInterfaceProps> = ({ onEnd }) => {
  const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
  const [answer, setAnswer] = useState('');
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [totalScore, setTotalScore] = useState(0);
  const [questionsAnswered, setQuestionsAnswered] = useState(0);

  const fetchNextQuestion = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/next-question', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          previousScore: result?.score,
          previousTopic: currentQuestion?.topic,
          previousDifficulty: currentQuestion?.difficulty
        })
      });
      
      if (!response.ok) throw new Error('Failed to fetch next question');
      
      const question = await response.json();
      setCurrentQuestion(question);
      setResult(null);
      setAnswer('');
    } catch (err) {
      setError('Failed to load next question. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (!answer.trim()) return;
    
    setIsEvaluating(true);
    try {
      const response = await fetch('/api/evaluate-answer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          questionId: currentQuestion?.id,
          answer: answer.trim()
        })
      });
      
      if (!response.ok) throw new Error('Evaluation failed');
      
      const evaluationResult = await response.json();
      setResult(evaluationResult);
      setTotalScore(prev => prev + evaluationResult.score);
      setQuestionsAnswered(prev => prev + 1);
    } catch (err) {
      setError('Failed to evaluate answer. Please try again.');
    } finally {
      setIsEvaluating(false);
    }
  };

  useEffect(() => {
    fetchNextQuestion();
  }, []);

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  return (
    <Box>
      <StyledPaper elevation={2}>
        <Stack direction="row" spacing={2} mb={2}>
          <Chip 
            label={`Topic: ${currentQuestion?.topic}`} 
            color="primary" 
            variant="outlined" 
          />
          <Chip 
            label={`Difficulty: ${currentQuestion?.difficulty}/10`} 
            color="secondary" 
            variant="outlined" 
          />
          <Chip 
            label={`Score: ${totalScore}/${questionsAnswered * 10}`} 
            color="info" 
            variant="outlined" 
          />
        </Stack>

        <Typography variant="h6" gutterBottom>
          Question {questionsAnswered + 1}
        </Typography>
        
        <Box mb={3}>
          <ReactMarkdown>{currentQuestion?.content || ''}</ReactMarkdown>
        </Box>

        <AnswerBox
          fullWidth
          multiline
          rows={6}
          variant="outlined"
          placeholder="Type your answer here..."
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          disabled={isEvaluating || !!result}
        />

        <Stack direction="row" spacing={2} justifyContent="flex-end">
          <Button
            variant="contained"
            color="primary"
            onClick={handleSubmit}
            disabled={isEvaluating || !answer.trim() || !!result}
          >
            {isEvaluating ? <CircularProgress size={24} /> : 'Submit'}
          </Button>
          
          {result && (
            <Button
              variant="contained"
              color="secondary"
              onClick={fetchNextQuestion}
            >
              Next Question
            </Button>
          )}
          
          <Button
            variant="outlined"
            color="error"
            onClick={onEnd}
          >
            End Interview
          </Button>
        </Stack>
      </StyledPaper>

      {result && (
        <ResultCard>
          <CardContent>
            <Typography variant="h6" gutterBottom color={result.score >= 7 ? 'success.main' : 'error.main'}>
              Score: {result.score}/10
            </Typography>
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle1" gutterBottom>
              Feedback:
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              {result.feedback}
            </Typography>
            {result.correctAnswer && (
              <>
                <Typography variant="subtitle1" gutterBottom>
                  Sample Solution:
                </Typography>
                <Box sx={{ backgroundColor: 'grey.100', p: 2, borderRadius: 1 }}>
                  <ReactMarkdown>{result.correctAnswer}</ReactMarkdown>
                </Box>
              </>
            )}
          </CardContent>
        </ResultCard>
      )}
    </Box>
  );
}; 