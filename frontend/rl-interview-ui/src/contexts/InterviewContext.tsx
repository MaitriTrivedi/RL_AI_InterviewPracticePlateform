import React, { createContext, useContext, useState, type ReactNode } from 'react';
import { Interview } from '../types';

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

interface InterviewContextType {
  userId: string | null;
  setUserId: (id: string | null) => void;
}

const InterviewContext = createContext<InterviewContextType>({
  userId: null,
  setUserId: () => {},
});

export const useInterview = () => useContext(InterviewContext);

export const InterviewProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [userId, setUserId] = useState<string | null>(null);

  return (
    <InterviewContext.Provider value={{ userId, setUserId }}>
      {children}
    </InterviewContext.Provider>
  );
};

export default InterviewContext; 