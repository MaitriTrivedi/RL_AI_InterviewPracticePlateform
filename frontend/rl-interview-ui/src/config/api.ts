// API Configuration
const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export const API_ENDPOINTS = {
  UPLOAD_RESUME: `${BASE_URL}/resume/upload`,
  GET_RESUME: (resumeId: string) => `${BASE_URL}/resume/${resumeId}`,
  CREATE_INTERVIEW: `${BASE_URL}/interview/start`,
  NEXT_QUESTION: (interviewId: string) => `${BASE_URL}/interview/${interviewId}/next-question`,
  SUBMIT_ANSWER: (interviewId: string) => `${BASE_URL}/interview/${interviewId}/submit-answer`,
  END_INTERVIEW: (interviewId: string) => `${BASE_URL}/interview/${interviewId}/end`,
  GET_RESULTS: (interviewId: string) => `${BASE_URL}/interview/${interviewId}/results`,
};

export default API_ENDPOINTS; 