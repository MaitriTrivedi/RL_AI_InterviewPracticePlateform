import React, { createContext, useContext, useState, ReactNode } from 'react';

interface Question {
  questionId: string;
  question: string;
  difficulty: number;
}

interface Answer {
  questionId: string;
  answer: string;
  score: number;
  feedback?: {
    correctness: { score: number; comment: string };
    efficiency: { score: number; comment: string };
    code_quality: { score: number; comment: string };
    understanding: { score: number; comment: string };
  };
  improvement_suggestions?: string[];
  overall_feedback?: string;
}

interface Interview {
  interviewId: string;
  topic: string;
  current_question: Question | null;
  questions: Question[];
  answers: Answer[];
  status: 'in_progress' | 'completed';
}

interface InterviewContextType {
  currentInterview: Interview | null;
  setCurrentInterview: (interview: Interview | null) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
  error: string | null;
  setError: (error: string | null) => void;
}

const InterviewContext = createContext<InterviewContextType | undefined>(undefined);

export const InterviewProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentInterview, setCurrentInterview] = useState<Interview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <InterviewContext.Provider
      value={{
        currentInterview,
        setCurrentInterview,
        loading,
        setLoading,
        error,
        setError,
      }}
    >
      {children}
    </InterviewContext.Provider>
  );
};

export const useInterview = () => {
  const context = useContext(InterviewContext);
  if (context === undefined) {
    throw new Error('useInterview must be used within an InterviewProvider');
  }
  return context;
};

export default InterviewContext; 