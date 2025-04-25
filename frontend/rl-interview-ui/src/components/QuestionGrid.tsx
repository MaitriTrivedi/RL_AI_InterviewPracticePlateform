import React from 'react';
import { Box, Card, CardContent, Typography, Button, Grid } from '@mui/material';
import { styled } from '@mui/material/styles';

const QuestionCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  cursor: 'pointer',
  transition: 'transform 0.2s ease-in-out',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: theme.shadows[4],
  },
}));

interface Question {
  questionId: string;
  question: string;
  difficulty: number;
}

interface QuestionGridProps {
  questions: Question[];
  onSelectQuestion: (questionId: string) => void;
  selectedQuestionId?: string;
}

const QuestionGrid: React.FC<QuestionGridProps> = ({
  questions,
  onSelectQuestion,
  selectedQuestionId,
}) => {
  // Function to extract the actual question text from the Gemini response
  const extractQuestionText = (fullText: string) => {
    const questionMatch = fullText.match(/Question:\s*(.*?)(?=Expected Time:|$)/s);
    return questionMatch ? questionMatch[1].trim() : fullText;
  };

  const getDifficultyColor = (difficulty: number) => {
    if (difficulty <= 4) return '#4caf50'; // Easy - Green
    if (difficulty <= 6) return '#ff9800'; // Medium - Orange
    return '#f44336'; // Hard - Red
  };

  if (!questions || questions.length === 0) {
    return (
      <Box sx={{ flexGrow: 1, p: 3 }}>
        <Typography variant="h6" color="text.secondary" align="center">
          No questions available
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Grid container spacing={3}>
        {questions.map((question, index) => (
          <Grid item xs={12} md={6} lg={4} key={question.questionId}>
            <QuestionCard 
              onClick={() => onSelectQuestion(question.questionId)}
              sx={{
                bgcolor: selectedQuestionId === question.questionId ? 'action.selected' : 'background.paper',
              }}
            >
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Question {index + 1}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Typography 
                    variant="caption" 
                    sx={{ 
                      color: getDifficultyColor(question.difficulty),
                      fontWeight: 'bold',
                    }}
                  >
                    Difficulty: {question.difficulty}/10
                  </Typography>
                </Box>
                <Typography 
                  variant="body2" 
                  sx={{ 
                    mb: 2,
                    height: '100px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    display: '-webkit-box',
                    WebkitLineClamp: 4,
                    WebkitBoxOrient: 'vertical',
                  }}
                >
                  {extractQuestionText(question.question)}
                </Typography>
                <Button 
                  variant={selectedQuestionId === question.questionId ? "contained" : "outlined"}
                  fullWidth 
                  sx={{ mt: 'auto' }}
                >
                  {selectedQuestionId === question.questionId ? "Selected" : "Select Question"}
                </Button>
              </CardContent>
            </QuestionCard>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default QuestionGrid; 