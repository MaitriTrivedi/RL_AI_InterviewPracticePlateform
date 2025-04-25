// Resume types
export interface Education {
  school: string;
  degree: string;
  field: string;
  start_date: string;
  end_date: string;
}

export interface WorkExperience {
  company: string;
  title: string;
  start_date: string;
  end_date: string;
  description: string[];
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
  skills: string[];
}

// Interview types
export interface Question {
  questionId: string;
  question: string;
  difficulty: string;
  topic: string;
}

export interface Answer {
  answer: string;
  score?: number;
  feedback?: string;
}

export interface Interview {
  interviewId: string;
  topic: string;
  difficulty: number;
  maxQuestions: number;
  status: 'pending' | 'in_progress' | 'completed';
  questions: Question[];
  answers: Answer[];
  currentQuestionIdx: number;
}

// App state type
export interface AppState {
  loading: boolean;
  error: string | null;
  currentInterview: Interview | null;
  currentResume: ResumeData | null;
  resumeId: string | null;
  interviewHistory: Interview[];
} 