import axios from 'axios';

// Base URL for our API
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Create axios instance with base URL
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

const API_BASE_URL = 'http://localhost:5000/api';

export interface ResumeData {
  education: Array<{
    institution: string;
    degree: string;
    year: string;
    gpa?: string;
  }>;
  work_experience: Array<{
    company: string;
    year: string;
    description: string;
  }>;
  projects: Array<{
    name: string;
    description: string;
    technologies: string[];
  }>;
}

export interface Question {
  questionId: string;
  question: string;
  difficulty: string;
}

export interface Answer {
  questionId: string;
  answer: string;
  score: number;
}

export interface Interview {
  interviewId: string;
  topic: string;
  difficulty: number;
  maxQuestions: number;
  currentQuestionIdx: number;
  questions: Question[];
  answers: Answer[];
  status: 'in_progress' | 'completed';
}

// Resume API
export const resumeApi = {
  uploadResume: async (file: File) => {
    const formData = new FormData();
    formData.append('resume', file);
    
    const response = await axios.post(`${API_BASE_URL}/resume/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response;
  },
  
  getResume: async (resumeId: string) => {
    const response = await axios.get(`${API_BASE_URL}/resume/${resumeId}`);
    return response;
  },
};

// Question API
export const questionApi = {
  generateQuestions: (params: { 
    resumeId?: string, 
    topic: string, 
    difficulty: number, 
    numQuestions?: number 
  }) => {
    return api.post('/questions/generate', params);
  },
  evaluateAnswer: (params: { 
    questionId: string, 
    answer: string,
    topic: string
  }) => {
    return api.post('/questions/evaluate', params);
  }
};

// RL Agent API
export const rlAgentApi = {
  createInterview: async (topic: string, difficulty: number = 5, maxQuestions: number = 10) => {
    const response = await axios.post(`${API_BASE_URL}/interview/new`, {
      topic,
      difficulty,
      maxQuestions,
    });
    return response;
  },
  
  getNextQuestion: async (interviewId: string) => {
    const response = await axios.get(`${API_BASE_URL}/interview/${interviewId}/next-question`);
    return response;
  },
  
  submitAnswer: async (interviewId: string, questionId: string, answer: string) => {
    const response = await axios.post(`${API_BASE_URL}/interview/${interviewId}/submit-answer`, {
      questionId,
      answer,
    });
    return response;
  },
  
  getResults: async (interviewId: string) => {
    const response = await axios.get(`${API_BASE_URL}/interview/${interviewId}/results`);
    return response;
  },
};

export default api; 