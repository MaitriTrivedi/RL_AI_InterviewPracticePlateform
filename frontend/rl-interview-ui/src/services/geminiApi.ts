import axios from 'axios';

const GEMINI_API_KEY = process.env.REACT_APP_GEMINI_API_KEY;
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent';

interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{
        text: string;
      }>;
    };
  }>;
}

export const geminiApi = {
  generateResponse: async (prompt: string): Promise<string> => {
    try {
      if (!GEMINI_API_KEY) {
        throw new Error('Gemini API key not found in environment variables');
      }

      const response = await axios.post<GeminiResponse>(
        `${GEMINI_API_URL}?key=${GEMINI_API_KEY}`,
        {
          contents: [
            {
              parts: [
                {
                  text: prompt
                }
              ]
            }
          ]
        }
      );

      const generatedText = response.data.candidates[0]?.content.parts[0]?.text;
      if (!generatedText) {
        throw new Error('No response generated from Gemini API');
      }

      return generatedText;
    } catch (error: any) {
      console.error('Error calling Gemini API:', error);
      throw new Error(error.response?.data?.error?.message || error.message || 'Failed to generate response');
    }
  }
}; 