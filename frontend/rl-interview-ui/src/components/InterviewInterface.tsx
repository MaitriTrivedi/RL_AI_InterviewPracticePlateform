import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
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
import { useAppContext } from '../contexts/AppContext';
import { API_ENDPOINTS } from '../config/api';
import { Question as AppQuestion } from '../types';

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

interface Question extends AppQuestion {
  content: string;
  follow_up_questions: string[];
  subtopic: string;
  session_stats: {
    questions_asked: number;
    average_performance: number;
    current_topic: string;
  };
}

interface EvaluationResult {
  score: number;
  feedback: string;
  correctAnswer?: string;
}

interface RawEvaluationResponse {
  score: number;
  feedback: string;
  strengths: string[];
  improvements: string[];
}

interface EvaluationResponse {
  evaluation: {
    score: number;
    overall_feedback: string;
    correct_answer?: string;
    strengths: string[];
    improvements: string[];
  };
  error?: string;
}

interface AnswerSubmissionRequest {
  question_id: string;
  answer: string;
  session_id: string;
}

export const InterviewInterface: React.FC<InterviewInterfaceProps> = ({ onEnd }): JSX.Element => {
  const navigate = useNavigate();
  const { state: { currentInterview }, setCurrentInterview, setError: setAppError } = useAppContext();
  const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
  const [answer, setAnswer] = useState('');
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [totalScore, setTotalScore] = useState(0);
  const [questionsAnswered, setQuestionsAnswered] = useState(0);

  useEffect(() => {
    if (!currentInterview?.interviewId) {
      navigate('/');
      return;
    }
    fetchNextQuestion();
  }, [currentInterview?.interviewId, navigate]);

  const fetchNextQuestion = async () => {
    if (!currentInterview?.interviewId) return;
    
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/next-question', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: currentInterview.interviewId,
          question: {
            time_allocated: 15,
            topic_index: 0
          }
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch next question');
      }
      
      const question = await response.json();
      setCurrentQuestion(question);
      setResult(null);
      setAnswer('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load next question. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (!answer.trim() || !currentInterview?.interviewId || !currentQuestion?.questionId) return;
    
    setIsEvaluating(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_ENDPOINTS.SUBMIT_ANSWER(currentInterview.interviewId)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          question_id: currentQuestion.questionId,
          answer: answer.trim(),
          session_id: currentInterview.interviewId
        })
      });

      const responseText = await response.text();
      console.log('Raw response text:', responseText);

      // Remove Markdown code block formatting if present
      const cleanResponseText = responseText
        .replace(/^```json\n/, '')  // Remove opening ```json
        .replace(/\n```$/, '')      // Remove closing ```
        .trim();

      let responseData: EvaluationResponse;
      try {
        if (!cleanResponseText) {
          throw new Error('Empty response from server');
        }
        const rawResponse: RawEvaluationResponse = JSON.parse(cleanResponseText);
        console.log('Parsed response data:', rawResponse);

        responseData = {
          evaluation: {
            score: rawResponse.score || 0,
            overall_feedback: rawResponse.feedback || 'No feedback provided',
            correct_answer: undefined,
            strengths: rawResponse.strengths || [],
            improvements: rawResponse.improvements || []
          }
        };

        setResult({
          score: responseData.evaluation.score,
          feedback: `${responseData.evaluation.overall_feedback}\n\nStrengths:\n${responseData.evaluation.strengths.map(s => `• ${s}`).join('\n')}\n\nPossible Improvements:\n${responseData.evaluation.improvements.map(i => `• ${i}`).join('\n')}`,
          correctAnswer: responseData.evaluation.correct_answer
        });

        setTotalScore(prev => prev + responseData.evaluation.score);
        setQuestionsAnswered(prev => prev + 1);
      } catch (error) {
        console.error('JSON parse error:', error);
        throw new Error('Failed to parse server response');
      }
    } catch (error) {
      console.error('Error in submit handler:', error);
      setError(error instanceof Error ? error.message : 'Failed to submit answer');
    } finally {
      setIsEvaluating(false);
    }
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button variant="contained" onClick={() => navigate('/')}>
          Return to Home
        </Button>
      </Box>
    );
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
          <ReactMarkdown>{currentQuestion?.question || ''}</ReactMarkdown>
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
              onClick={() => navigate('/next-question')}
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