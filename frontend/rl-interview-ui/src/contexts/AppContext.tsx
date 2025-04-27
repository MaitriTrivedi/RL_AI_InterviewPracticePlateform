import React, { createContext, useContext, useState, ReactNode } from 'react';
import { AppState, ResumeData, Interview } from '../types/index';

// Create initial state
const initialState: AppState = {
  loading: false,
  error: null,
  currentInterview: null,
  currentResume: null,
  resumeId: null,
  interviewHistory: [],
};

// Create the context
const AppContext = createContext<{
  state: AppState;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setCurrentInterview: (interview: Interview | null) => void;
  setCurrentResume: (resume: ResumeData | null) => void;
  setResumeId: (resumeId: string | null) => void;
  addInterviewToHistory: (interview: Interview) => void;
}>({
  state: initialState,
  setLoading: () => {},
  setError: () => {},
  setCurrentInterview: () => {},
  setCurrentResume: () => {},
  setResumeId: () => {},
  addInterviewToHistory: () => {},
});

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, setState] = useState<AppState>(initialState);

  const setLoading = (loading: boolean) => {
    setState(prev => ({ ...prev, loading }));
  };

  const setError = (error: string | null) => {
    setState(prev => ({ ...prev, error }));
  };

  const setCurrentInterview = (interview: Interview | null) => {
    setState(prev => ({ ...prev, currentInterview: interview }));
  };

  const setCurrentResume = (resume: ResumeData | null) => {
    setState(prev => ({ ...prev, currentResume: resume }));
  };

  const setResumeId = (resumeId: string | null) => {
    setState(prev => ({ ...prev, resumeId }));
  };

  const addInterviewToHistory = (interview: Interview) => {
    setState(prev => ({
      ...prev,
      interviewHistory: [...prev.interviewHistory, interview],
    }));
  };

  return (
    <AppContext.Provider
      value={{
        state,
        setLoading,
        setError,
        setCurrentInterview,
        setCurrentResume,
        setResumeId,
        addInterviewToHistory,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export default AppContext; 