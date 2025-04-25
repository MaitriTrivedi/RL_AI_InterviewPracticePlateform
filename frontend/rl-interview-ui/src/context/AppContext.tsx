import React, { createContext, useContext, useState, ReactNode } from 'react';
import { AppState, ResumeData, Interview } from '../types';

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

// Create provider component
export const AppProvider = ({ children }: { children: ReactNode }) => {
  const [state, setState] = useState<AppState>(initialState);

  const setLoading = (loading: boolean) => {
    setState(prevState => ({ ...prevState, loading }));
  };

  const setError = (error: string | null) => {
    setState(prevState => ({ ...prevState, error }));
  };

  const setCurrentInterview = (currentInterview: Interview | null) => {
    setState(prevState => ({ ...prevState, currentInterview }));
  };

  const setCurrentResume = (currentResume: ResumeData | null) => {
    setState(prevState => ({ ...prevState, currentResume }));
  };

  const setResumeId = (resumeId: string | null) => {
    setState(prevState => ({ ...prevState, resumeId }));
  };

  const addInterviewToHistory = (interview: Interview) => {
    setState(prevState => ({
      ...prevState,
      interviewHistory: [...prevState.interviewHistory, interview],
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

// Create a custom hook for using the context
export const useAppContext = () => useContext(AppContext);

export default AppContext; 