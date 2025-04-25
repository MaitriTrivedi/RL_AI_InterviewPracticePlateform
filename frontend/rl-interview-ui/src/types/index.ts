// Resume Types
export interface Education {
  institution: string;
  degree: string;
  year: string;
  gpa?: string;
}

export interface WorkExperience {
  company: string;
  year: string;
  description: string;
}

export interface Project {
  name: string;
  description: string;
  technologies: string[];
}

export interface ResumeData {
  education: Education[];
  work_experience: WorkExperience[];
  projects: Project[];
}

// Question Types
export interface Question {
  questionId?: string;
  question: string;
  difficulty: string;
  questionNo: string;
}

export interface Answer {
  questionId: string;
  answer: string;
  score?: number;
  feedback?: string;
}

// Interview Types
export interface Interview {
  interviewId: string;
  resumeId: string;
  topic: string;
  currentQuestionIdx: number;
  maxQuestions: number;
  questions: Question[];
  answers: Answer[];
  status: 'in_progress' | 'completed';
  averageScore?: number;
}

// User Interface Types
export interface UploadResumeFormData {
  file: File | null;
}

export interface StartInterviewFormData {
  resumeId: string;
  topic?: string;
  difficulty?: number;
  maxQuestions?: number;
}

export interface QuestionFormData {
  answer: string;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  error?: string;
}

// Application State
export interface AppState {
  loading: boolean;
  error: string | null;
  currentInterview: Interview | null;
  currentResume: ResumeData | null;
  resumeId: string | null;
  interviewHistory: Interview[];
} 