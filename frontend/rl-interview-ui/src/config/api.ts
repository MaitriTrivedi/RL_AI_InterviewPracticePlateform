// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export const API_ENDPOINTS = {
  UPLOAD_RESUME: `${API_BASE_URL}/resume/upload`,
  GET_RESUME: (resumeId: string) => `${API_BASE_URL}/resume/${resumeId}`,
  CREATE_INTERVIEW: `${API_BASE_URL}/interview/start`,
  NEXT_QUESTION: (interviewId: string) => `${API_BASE_URL}/interview/${interviewId}/next-question`,
  SUBMIT_ANSWER: (interviewId: string) => `${API_BASE_URL}/interview/${interviewId}/submit-answer`,
  END_INTERVIEW: (interviewId: string) => `${API_BASE_URL}/interview/end`,
  GET_RESULTS: (interviewId: string) => `${API_BASE_URL}/interview/${interviewId}/results`,
};

export default API_ENDPOINTS; 