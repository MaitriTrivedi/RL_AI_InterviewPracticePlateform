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
  questionId: string;
  topic: string;
  subtopic: string;
  difficulty: number;
  content: string;
  follow_up_questions: string[];
  evaluation_points: string[];
  expected_time?: number;
}

export interface Answer {
  questionId: string;
  answer: string;
  score?: number;
  feedback?: {
    [criterion: string]: {
      score: number;
      comment: string;
    };
  };
  improvement_suggestions?: string[];
  overall_feedback?: string;
}

// Add Evaluation interface before InterviewState
export interface Evaluation {
  score: number;
  feedback: string;
  strengths: string[];
  improvements: string[];
}

// Interview Types
export interface SessionStats {
  questions_asked: number;
  average_performance: number;
  current_topic: string;
  current_subtopic: string;
  current_difficulty: number;
  topic_performances: Record<string, number>;
  learning_progress: Record<string, number>;
}

export interface InterviewState {
  sessionId: string;
  currentQuestion: Question;
  sessionStats: SessionStats;
  answers: Answer[];
  evaluations: Evaluation[];
  isComplete: boolean;
}

export interface Interview {
  interviewId: string;
  userId: string;
  currentQuestion: Question;
  currentQuestionIdx: number;
  questions: Question[];
  answers: Answer[];
  status: 'in_progress' | 'completed';
  difficulty: number;
  stats: SessionStats;
}

export interface SubmitAnswerResponse {
  evaluation: {
    score: number;
    feedback: string;
    strengths: string[];
    improvements: string[];
  };
  current_state: {
    current_question: Question;
    session_stats: SessionStats;
  };
  next_state: {
    next_question: Question;
    next_difficulty: number;
    next_topic: string;
    next_subtopic: string;
    interview_complete: boolean;
  };
}

export interface FinalStats {
  questions_asked: number;
  average_performance: number;
  final_difficulty: number;
  topics_covered: string[];
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

export interface AppQuestion {
  id: string;
  topic: string;
  subtopic: string;
  difficulty: number;
  content: string;
  follow_up_questions: string[];
  evaluation_points: string[];
  expected_time: number;
} 