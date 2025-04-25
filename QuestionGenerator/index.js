// gemini_qa.js
import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from "@google/generative-ai";
import readline from 'readline';
import dotenv from 'dotenv';

dotenv.config(); // Load environment variables from .env file

// --- Configuration ---
const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
    console.error("ERROR: GEMINI_API_KEY environment variable not set.");
    console.error("Please create a .env file with GEMINI_API_KEY=YOUR_KEY or set the environment variable.");
    process.exit(1);
}

// Choose the model
const modelName = "gemini-1.5-flash-latest"; // Or "gemini-pro"

// Configure safety settings (optional, adjust as needed)
const safetySettings = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
];

const genAI = new GoogleGenerativeAI(apiKey);
let model;
try {
    model = genAI.getGenerativeModel({ model: modelName, safetySettings });
    console.log(`Using model: ${modelName}`);
} catch (error) {
    console.error(`Error initializing the Generative Model '${modelName}':`, error.message);
    process.exit(1);
}

// --- Readline Setup for User Input ---
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

// Helper function to ask questions using promises
function askQuestion(query) {
    return new Promise((resolve) => rl.question(query, resolve));
}

// --- Helper Functions ---

function cleanJsonResponse(text) {
    if (!text || typeof text !== 'string') return text;
    const match = text.match(/```json\s*([\s\S]*?)\s*```/);
    if (match && match[1]) {
        return match[1].trim();
    }
    const cleanedText = text.trim();
    if ((cleanedText.startsWith('[') && cleanedText.endsWith(']')) ||
        (cleanedText.startsWith('{') && cleanedText.endsWith('}'))) {
        return cleanedText;
    }
    return cleanedText;
}

async function callGeminiChat(chatSession, prompt) {
    try {
        const result = await chatSession.sendMessage(prompt);
        const response = result.response;
        const responseText = response.text();

        if (response.promptFeedback?.blockReason) {
            console.error(`\n--- ERROR: Prompt Blocked by API ---`);
            console.error(`Reason: ${response.promptFeedback.blockReason}`);
            // Log ratings if available
            response.promptFeedback.safetyRatings?.forEach(rating => {
                console.error(`  ${rating.category}: ${rating.probability}`);
            });
            return null;
        }
        if (response.candidates?.[0]?.finishReason && response.candidates[0].finishReason !== 'STOP') {
            console.warn(`\n--- Warning: Response Terminated Abnormally ---`);
            console.warn(`Reason: ${response.candidates[0].finishReason}`);
            // Log ratings if available
            response.candidates[0].safetyRatings?.forEach(rating => {
                console.warn(`  ${rating.category}: ${rating.probability}`);
            });
            return responseText; // Still return text if available
        }
        return responseText;
    } catch (error) {
        console.error(`\n--- An API Error Occurred During Chat ---`);
        console.error(`Error details:`, error.message);
        return null;
    }
}

function parseJsonResponse(responseText, expectedTypeConstructor) {
    if (responseText === null || responseText === undefined) {
        console.log("Debug: Cannot parse JSON, input responseText is null/undefined.");
        return null;
    }
    const cleanedText = cleanJsonResponse(responseText);
    try {
        const data = JSON.parse(cleanedText);
        if (expectedTypeConstructor === Array && Array.isArray(data)) {
            return data;
        } else if (expectedTypeConstructor === Object && typeof data === 'object' && data !== null && !Array.isArray(data)) {
            return data;
        } else {
            console.error(`\n--- Error: Unexpected JSON structure ---`);
            console.error(`Expected type: ${expectedTypeConstructor.name}, Got type: ${Array.isArray(data) ? 'Array' : typeof data}`);
            console.error(`Received data structure:`, data);
            console.error(`Original Response Text (before cleaning):\n${responseText}`);
            return null;
        }
    } catch (error) {
        if (error instanceof SyntaxError) {
            console.error(`\n--- Error: Failed to decode JSON (SyntaxError) ---`);
            console.error(`Error message: ${error.message}`);
            console.error(`Text attempted to parse (after cleaning):\n${cleanedText}`);
            console.error(`Original Response Text (before cleaning):\n${responseText}`);
        } else {
            console.error(`\n--- An unexpected error occurred during JSON parsing ---`);
            console.error(`Error details:`, error);
        }
        return null;
    }
}


// --- Main Application Logic ---

async function run() {
    console.log("Welcome to the Gemini Q&A Evaluator!");
    console.log("=".repeat(40));

    let topic = "";
    let chatSession = null;

    // --- 1. Initialize Chat / Get Topic ---
    while (!topic) {
        topic = (await askQuestion("Enter the topic for the Q&A session (e.g., Docker, Java, Kubernetes): ")).trim();
        if (!topic) console.log("Topic cannot be empty.");
    }

    console.log(`\nInitializing chat session for topic: '${topic}'...`);
    try {
        const initialHistory = [
            { role: "user", parts: [{ text: `We are starting an interactive Q&A session about '${topic}'. I will ask you to generate technical questions around a specific difficulty score (0-10). I will then choose one, provide an answer, and you will evaluate my answer based on technical accuracy and key points, providing a score out of 10 (allowing decimals). Please remember the topic and the questions already asked throughout our conversation to avoid repetition. Respond strictly in the JSON formats requested.` }] },
            { role: "model", parts: [{ text: `Understood. I am ready to begin the Q&A session on '${topic}'. I will generate unique questions targeting the difficulty score you provide, wait for your answer to a chosen question, and then evaluate it based on technical accuracy and key points, providing a score out of 10 (including decimals) in the requested JSON format.` }] }
        ];
        chatSession = model.startChat({ history: initialHistory });
        console.log(`Chat session started for '${topic}'.`);
    } catch (error) {
        console.error(`Error starting chat session:`, error.message);
        rl.close();
        process.exit(1);
    }


    // --- Main Loop ---
    while (true) {
        // --- 2. Get Difficulty Score ---
        let difficultyScore = null; // Use null to indicate not yet set
        while (difficultyScore === null) {
            const input = (await askQuestion(`\nEnter desired difficulty score for '${topic}' questions (0.0 - 10.0): `)).trim();
            const score = parseFloat(input);
            if (!isNaN(score) && score >= 0 && score <= 10) {
                difficultyScore = score; // Store the valid score
            } else {
                console.log("Invalid input. Please enter a number between 0.0 and 10.0.");
            }
        }
        // Format for prompt consistency if needed, though number is fine
        const difficultyScoreStr = difficultyScore.toFixed(1);

        // --- 3. Generate Questions ---
        console.log(`\nGenerating questions for '${topic}' targeting difficulty score ~${difficultyScoreStr}...`);
        // ***** MODIFIED PROMPT TO USE NUMERICAL DIFFICULTY *****
        const promptQuestions = `
        Generate exactly 5 new, unique technical questions about the topic '${topic}' based on our ongoing conversation.
        Target difficulty score: Aim for questions around ${difficultyScoreStr} on a scale of 0.0 (very easy) to 10.0 (expert).

        For each generated question:
        1.  Provide the specific question text.
        2.  Estimate *its* specific difficulty score on the 0.0 to 10.0 scale, formatted to one decimal place.
        3.  Assign a sequential question number for this batch, starting from 1.

        IMPORTANT: Ensure these questions are different from any previously asked in this chat session. Do not include introductory text.

        Format the output STRICTLY as a JSON list of objects, like this example:
        [
            {"question": "What is RAM?", "difficulty": "1.5", "questionNo": "1"},
            {"question": "Explain memory virtualization.", "difficulty": "7.8", "questionNo": "2"},
            {"question": "...", "difficulty": "...", "questionNo": "3"},
            {"question": "...", "difficulty": "...", "questionNo": "4"},
            {"question": "...", "difficulty": "...", "questionNo": "5"}
        ]
        The output MUST be ONLY the valid JSON list and nothing else.
        `;

        const responseTextQuestions = await callGeminiChat(chatSession, promptQuestions);
        const questionsDataRaw = parseJsonResponse(responseTextQuestions, Array);

        if (!questionsDataRaw) {
            console.log("Could not get valid questions from Gemini. Trying again.");
            continue;
        }

        // --- Validate and Filter Questions ---
        const validQuestions = [];
        let malformedFound = false;
        for (let i = 0; i < questionsDataRaw.length; i++) {
            const q = questionsDataRaw[i];
            if (q && typeof q === 'object' && q.question && q.difficulty && q.questionNo) {
                try {
                    const qText = String(q.question).trim();
                    const qDiffStr = String(q.difficulty);
                    const qNumStr = String(q.questionNo);
                    // Validate difficulty and number format
                    const qDiffNum = parseFloat(qDiffStr);
                    const qNumInt = parseInt(qNumStr);

                    if (!qText) throw new Error("Empty question text");
                    if (isNaN(qDiffNum) || qDiffNum < 0 || qDiffNum > 10) throw new Error("Invalid difficulty format/range");
                    if (isNaN(qNumInt)) throw new Error("Invalid question number format");

                    validQuestions.push({ question: qText, difficulty: qDiffNum.toFixed(1), questionNo: qNumStr }); // Store difficulty formatted
                } catch (e) {
                    console.warn(`Warning: Skipping question ${i + 1} due to invalid format/type/range (${e.message}):`, q);
                    malformedFound = true;
                }
            } else {
                console.warn(`Warning: Skipping question ${i + 1} due to missing keys or incorrect type:`, q);
                malformedFound = true;
            }
        }

        if (validQuestions.length === 0) {
            console.log("No valid questions were extracted from the response. Asking again.");
            continue;
        }
        if (malformedFound) {
            console.log("Note: Some potential questions returned by the API were skipped.");
        }

        const currentBatchQuestions = validQuestions.reduce((acc, q) => {
            acc[q.questionNo] = q;
            return acc;
        }, {});


        // --- 4. Display Questions ---
        console.log("\n--- Please choose a question to answer ---");
        validQuestions
            .sort((a, b) => parseInt(a.questionNo) - parseInt(b.questionNo))
            .forEach(qData => {
                console.log(`\n${qData.questionNo}. (Est. Difficulty: ${qData.difficulty})`); // Show estimated difficulty
                console.log(`   Q: ${qData.question}`);
            });

        // --- 5. User Selects and Answers ---
        let selectedQNum = null;
        let selectedQuestionData = null;
        while (selectedQNum === null) {
            const choice = (await askQuestion("\nEnter the number of the question you want to answer: ")).trim();
            if (currentBatchQuestions[choice]) {
                selectedQNum = choice;
                selectedQuestionData = currentBatchQuestions[choice];
            } else {
                console.log(`Invalid choice '${choice}'. Please enter one of: ${Object.keys(currentBatchQuestions).join(', ')}`);
            }
        }

        console.log(`\nYou selected Question ${selectedQNum}:`);
        console.log(`Q: ${selectedQuestionData.question}`);

        let userAnswer = "";
        while (!userAnswer) {
            userAnswer = (await askQuestion("Your Answer: ")).trim();
            if (!userAnswer) console.log("Answer cannot be empty.");
        }

        // --- 6. Evaluate Answer and Get Score ---
        console.log("\nEvaluating your answer (focusing on key points and technical terms)...");
        // ***** ADJUSTED SCORING PROMPT TO EMPHASIZE DECIMALS *****
        const promptScore = `
        You are an expert technical evaluator for the topic '${topic}'.
        I previously asked you for questions, and you provided a list. I chose to answer question number ${selectedQNum} from that list.

        The specific question was:
        "${selectedQuestionData.question}"

        My provided answer is:
        "${userAnswer}"

        **Evaluation Task:**
        1.  **Identify Key Technical Elements:** Based on the QUESTION, determine the essential concepts, terms, commands, etc., required for a correct answer.
        2.  **Analyze My Answer:** Examine MY ANSWER for the presence, accuracy, and appropriate use of these key technical elements.
        3.  **Score Based on Technical Accuracy:** Assign a score between 0.0 and 10.0, **allowing decimal values** (e.g., 7.5, 8.0, 9.2). This score should **primarily** reflect the technical accuracy, correct use of key terms, and understanding of core concepts.
            *   High (8.0-10.0): Strong understanding, accurate terms/concepts.
            *   Medium (5.0-7.9): Partial understanding, some correct elements but notable gaps/inaccuracies.
            *   Low (0.0-4.9): Misses main technical points, incorrect/irrelevant.
        4.  **Focus:** Prioritize technical substance and precision.

        **Output Format:**
        Respond ONLY with a valid JSON object containing the score, like this:
        {"score": "SCORE"}
        (e.g., {"score": "8.5"} or {"score": "7.0"} or {"score": "9"})
        Do not include any other text, just the JSON object.
        `;

        const responseTextScore = await callGeminiChat(chatSession, promptScore);
        const scoreData = parseJsonResponse(responseTextScore, Object);

        if (scoreData && scoreData.score !== undefined) {
            console.log(`\n--- Evaluation Complete ---`);
            const scoreValue = scoreData.score;
            const scoreNum = parseFloat(scoreValue);
            if (!isNaN(scoreNum)) {
                // Format to 1 decimal place for consistency
                console.log(`Score (based on key points/terms): ${scoreNum.toFixed(1)}/10.0`);
            } else {
                console.log(`Score (based on key points/terms): ${scoreValue}/10.0`); // Show raw if not number
            }
        } else {
            console.log("\n--- Could not get a valid score ---");
            console.log("Response might have been blocked, malformed, or missing the 'score' key.");
            if (responseTextScore) console.log(`Gemini's raw response text was:\n${responseTextScore}`);
            else console.log("(No response text received)");
        }


        // --- 7. Continue or Exit ---
        console.log("-".repeat(40));
        let nextAction = "";
        while (nextAction !== 'y' && nextAction !== 'n') {
            nextAction = (await askQuestion(`Ask another question on '${topic}' ('y') or exit ('n')? [y/n]: `)).toLowerCase().trim();
            if (nextAction !== 'y' && nextAction !== 'n') console.log("Please enter 'y' or 'n'.");
        }

        if (nextAction === 'n') {
            console.log("\nExiting the Q&A session. Goodbye!");
            break;
        }
        // Loop continues...
    }

    rl.close();
}

// --- Start the application ---
run().catch(error => {
    console.error("\nAn unexpected error occurred in the main execution:", error);
    rl.close();
    process.exit(1);
});