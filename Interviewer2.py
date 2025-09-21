import json
import random
import time
import os
from datetime import datetime
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('InterviewerAI.env')

class AIInterviewBot:
    def __init__(self, debug_mode=False):
        self.candidate_name = ""
        self.position = ""
        self.interview_type = ""
        self.responses = []
        self.interview_started = False
        self.model = None
        self.conversation_history = []
        self.asked_questions = set()  # Track asked questions to avoid duplicates
        self.question_categories = []  # Track question categories for variety
        self.debug_mode = debug_mode
        
        # Initialize Gemini API
        self.setup_gemini_api()
        
    def setup_gemini_api(self):
        """Setup Gemini API with your API key from .env file or environment"""
        # Get API key from environment variable (loaded from .env file)
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            print("🔑 No API key found in environment variables or .env file")
            print("💡 To set up your API key:")
            print("   1. Add GEMINI_API_KEY=your_key_here to InterviewerAI.env file")
            print("   2. Or set environment variable: GEMINI_API_KEY=your_key_here")
            print("   3. Get a free API key from: https://makersuite.google.com/app/apikey")
            print("\n🔄 Running in demo mode with fallback questions...")
            return
            
        try:
            print("🔌 Testing API connection...")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Test the API with a simple request
            test_response = self.model.generate_content("Say 'API test successful'")
            if test_response and test_response.text:
                print("✅ Gemini API connected successfully!")
                print(f"🔍 API Response Test: {test_response.text.strip()}")
                print("🎉 Full AI features enabled!")
            else:
                raise Exception("API test failed - no response received")
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error connecting to Gemini API: {error_msg}")
            
            if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
                print("🔑 API Key Issue Detected:")
                print("   • Your API key in .env file is invalid or expired")
                print("   • Check your InterviewerAI.env file")
                print("   • Get a new key from: https://makersuite.google.com/app/apikey")
                print("   • Make sure the key format is: GEMINI_API_KEY=your_key_here")
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                print("📊 Quota Issue Detected:")
                print("   • You may have exceeded your API usage limits")
                print("   • Check your Google AI Studio dashboard")
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                print("🌐 Network Issue Detected:")
                print("   • Check your internet connection")
                print("   • Try again in a few moments")
            else:
                print("🔧 General API Error:")
                print("   • Check your API key format in .env file")
                print("   • Ensure you have access to Gemini API")
            
            print("\n🔄 Running in demo mode with fallback questions...")
            self.model = None
            
    def generate_question(self, context: str = "") -> str:
        """Generate interview questions using Gemini AI"""
        if not self.model:
            # Enhanced fallback questions with more variety and categories
            fallback_questions = {
                "behavioral": [
                    "Tell me about a time when you had to work with a difficult team member.",
                    "Describe a situation where you had to meet a tight deadline.",
                    "Give me an example of when you had to learn something new quickly.",
                    "Tell me about a time you failed and what you learned from it.",
                    "Describe a situation where you had to give difficult feedback.",
                    "Tell me about a time you had to adapt to a major change at work.",
                    "Give me an example of when you went above and beyond for a project.",
                    "Describe a time when you had to resolve a conflict between team members."
                ],
                "technical": [
                    "Walk me through your approach to debugging a complex issue.",
                    "How do you stay updated with the latest technologies in your field?",
                    "Describe a technical challenge you solved recently.",
                    "What's your process for code review and quality assurance?",
                    "How do you approach performance optimization?",
                    "Tell me about a time you had to learn a new technology quickly.",
                    "Describe your experience with version control and collaboration tools.",
                    "How do you handle technical debt in your projects?"
                ],
                "general": [
                    "What motivates you in your career?",
                    "Where do you see yourself in 5 years?",
                    "What's your greatest professional achievement?",
                    "How do you handle stress and pressure?",
                    "What are your career goals?",
                    "How do you prioritize your work?",
                    "What's the most important thing you've learned recently?",
                    "How do you approach continuous learning?"
                ],
                "case_study": [
                    "How would you approach designing a scalable system?",
                    "Walk me through how you would optimize a slow-performing application.",
                    "How would you handle a situation where requirements change mid-project?",
                    "Describe your approach to data migration between systems.",
                    "How would you design a user authentication system?",
                    "What's your strategy for handling system downtime?",
                    "How would you approach testing a complex integration?",
                    "Describe how you would implement a new feature with minimal risk."
                ]
            }
            
            # Get questions for the current interview type
            questions = fallback_questions.get(self.interview_type, fallback_questions["general"])
            
            # Filter out already asked questions
            available_questions = [q for q in questions if q not in self.asked_questions]
            
            # If all questions have been asked, reset and add some variety
            if not available_questions:
                self.asked_questions.clear()
                available_questions = questions[:5]  # Take first 5 to avoid repetition
            
            selected_question = random.choice(available_questions)
            self.asked_questions.add(selected_question)
            return selected_question
            
        # Build comprehensive context including all previous questions
        previous_questions = [r['question'] for r in self.responses[-3:]]  # Last 3 questions
        previous_questions_text = "\n".join([f"- {q}" for q in previous_questions]) if previous_questions else "None"
        
        prompt = f"""
        You are an experienced interviewer conducting a {self.interview_type} interview for a {self.position} position.
        Candidate name: {self.candidate_name}
        
        Previous questions asked in this interview:
        {previous_questions_text}
        
        Current conversation context:
        {context}
        
        Generate ONE unique, thoughtful, professional interview question that:
        - Is appropriate for the {self.position} role and {self.interview_type} interview type
        - Follows best interview practices
        - Is completely different from the previous questions listed above
        - Builds on the conversation naturally
        - Encourages detailed, specific responses
        - Varies in difficulty and topic area
        
        Make sure the question is unique and hasn't been asked before.
        Return ONLY the question, no additional text or formatting.
        """
        
        try:
            if self.debug_mode:
                print(f"🔍 DEBUG: Sending prompt to Gemini API...")
                print(f"🔍 DEBUG: Prompt length: {len(prompt)} characters")
            
            response = self.model.generate_content(prompt)
            
            if self.debug_mode:
                print(f"🔍 DEBUG: Received response from API")
                print(f"🔍 DEBUG: Response type: {type(response)}")
                print(f"🔍 DEBUG: Response has text: {hasattr(response, 'text')}")
                if hasattr(response, 'text'):
                    print(f"🔍 DEBUG: Response text length: {len(response.text) if response.text else 0}")
            
            if not response or not response.text:
                raise Exception("Empty response from Gemini API")
                
            question = response.text.strip()
            
            # Clean up the question (remove quotes, extra whitespace, etc.)
            question = question.strip('"').strip("'").strip()
            
            if self.debug_mode:
                print(f"🔍 DEBUG: Cleaned question: '{question}'")
                print(f"🔍 DEBUG: Question length: {len(question)}")
            
            if not question or len(question) < 10:
                raise Exception("Generated question too short or empty")
            
            # Check for uniqueness
            if question in self.asked_questions:
                # Generate a follow-up question if duplicate
                follow_up_prompt = f"""
                The previous question was: "{question}"
                Generate a different, unique follow-up question for the same topic area.
                Return ONLY the new question, no additional text.
                """
                try:
                    follow_up_response = self.model.generate_content(follow_up_prompt)
                    if follow_up_response and follow_up_response.text:
                        question = follow_up_response.text.strip().strip('"').strip("'").strip()
                except Exception as follow_up_error:
                    print(f"⚠️  Follow-up generation failed: {follow_up_error}")
                    # Use original question if follow-up fails
            
            self.asked_questions.add(question)
            return question
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Error generating AI question: {error_msg}")
            
            # Provide specific error guidance
            if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
                print("🔑 Switching to demo mode due to API key issue...")
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                print("📊 API quota exceeded, using fallback questions...")
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                print("🌐 Network issue, using fallback questions...")
            else:
                print("🔧 Using fallback questions due to API error...")
            
            # Use fallback with uniqueness check
            fallback_questions = [
                "Tell me about a time when you overcame a significant challenge.",
                "Describe a project where you had to learn new skills quickly.",
                "How do you handle working under pressure?",
                "What's your greatest professional achievement?",
                "Tell me about a time you had to work with a difficult team member."
            ]
            
            for fallback in fallback_questions:
                if fallback not in self.asked_questions:
                    self.asked_questions.add(fallback)
                    return fallback
            
            # If all fallbacks used, reset and use first one
            self.asked_questions.clear()
            self.asked_questions.add(fallback_questions[0])
            return fallback_questions[0]
            
    def analyze_response(self, question: str, answer: str) -> Dict:
        """Analyze candidate's response using Gemini AI"""
        if not self.model:
            # Enhanced fallback analysis with variety
            fallback_analyses = [
                {
                    "feedback": "Great answer! You provided good detail and structure.",
                    "score": 8,
                    "strengths": ["Clear communication", "Good examples"],
                    "improvements": ["Consider adding more specific metrics or outcomes"],
                    "follow_up": None
                },
                {
                    "feedback": "Excellent response! You demonstrated strong problem-solving skills.",
                    "score": 7,
                    "strengths": ["Logical thinking", "Practical approach"],
                    "improvements": ["Try to quantify your impact where possible"],
                    "follow_up": None
                },
                {
                    "feedback": "Well done! Your answer shows good self-awareness and growth mindset.",
                    "score": 9,
                    "strengths": ["Self-reflection", "Learning orientation"],
                    "improvements": ["Consider adding more technical details"],
                    "follow_up": None
                }
            ]
            return random.choice(fallback_analyses)
            
        # Build context from previous responses for better analysis
        previous_context = ""
        if self.responses:
            recent_scores = [r.get('score', 0) for r in self.responses[-2:]]
            avg_recent_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0
            previous_context = f"\nPrevious responses average score: {avg_recent_score:.1f}/10"
            
        prompt = f"""
        As an expert interview coach, analyze this candidate's response:
        
        Position: {self.position}
        Interview Type: {self.interview_type}
        Question: {question}
        Answer: {answer}
        {previous_context}
        
        Provide analysis in this JSON format:
        {{
            "feedback": "Brief encouraging feedback (2-3 sentences) that's specific to their answer",
            "score": score_out_of_10,
            "strengths": ["strength1", "strength2"],
            "improvements": ["improvement1", "improvement2"],
            "follow_up": "relevant follow-up question or null"
        }}
        
        Be constructive, specific, and encouraging. Focus on interview skills, not just content.
        Consider the interview type and provide varied, personalized feedback.
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                raise Exception("Empty response from Gemini API")
                
            # Parse JSON response
            analysis_text = response.text.strip()
            # Remove markdown formatting if present
            if analysis_text.startswith('```json'):
                analysis_text = analysis_text[7:-3]
            elif analysis_text.startswith('```'):
                analysis_text = analysis_text[3:-3]
                
            analysis = json.loads(analysis_text)
            
            # Validate required fields
            required_fields = ["feedback", "score", "strengths", "improvements"]
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = "Not provided" if field == "feedback" else [] if field in ["strengths", "improvements"] else 0
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error in analysis: {e}")
            print("🔧 Using fallback analysis...")
            return self._get_fallback_analysis()
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Error analyzing response: {error_msg}")
            
            if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
                print("🔑 API key issue detected in analysis...")
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                print("📊 API quota exceeded in analysis...")
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                print("🌐 Network issue in analysis...")
            
            return self._get_fallback_analysis()
    
    def _get_fallback_analysis(self):
        """Get a fallback analysis when API fails"""
        fallback_analyses = [
            {
                "feedback": "Thank you for your detailed response! You provided good structure and examples.",
                "score": 8,
                "strengths": ["Clear communication", "Good examples"],
                "improvements": ["Consider adding more specific metrics or outcomes"],
                "follow_up": None
            },
            {
                "feedback": "Excellent response! You demonstrated strong problem-solving skills and clear thinking.",
                "score": 7,
                "strengths": ["Logical thinking", "Practical approach"],
                "improvements": ["Try to quantify your impact where possible"],
                "follow_up": None
            },
            {
                "feedback": "Well done! Your answer shows good self-awareness and a growth mindset.",
                "score": 9,
                "strengths": ["Self-reflection", "Learning orientation"],
                "improvements": ["Consider adding more technical details"],
                "follow_up": None
            }
        ]
        return random.choice(fallback_analyses)
    
    def generate_overall_analysis(self) -> Dict:
        """Generate comprehensive interview analysis"""
        if not self.model or not self.responses:
            return {"summary": "Interview completed successfully!"}
            
        # Prepare conversation summary
        conversation_summary = "\n".join([
            f"Q: {r['question']}\nA: {r['answer']}\nScore: {r.get('score', 'N/A')}"
            for r in self.responses[-5:]  # Last 5 questions for context
        ])
        
        prompt = f"""
        Analyze this complete interview performance for a {self.position} position:
        
        Candidate: {self.candidate_name}
        Interview Type: {self.interview_type}
        Questions Answered: {len(self.responses)}
        
        Recent Q&A History:
        {conversation_summary}
        
        Provide comprehensive analysis in JSON format:
        {{
            "overall_score": score_out_of_10,
            "summary": "Overall performance summary (3-4 sentences)",
            "top_strengths": ["strength1", "strength2", "strength3"],
            "areas_for_improvement": ["area1", "area2", "area3"],
            "specific_recommendations": ["rec1", "rec2", "rec3"],
            "interview_readiness": "Ready/Needs Practice/Needs Significant Work",
            "next_steps": ["step1", "step2"]
        }}
        
        Be honest but encouraging. Focus on actionable feedback.
        """
        
        try:
            response = self.model.generate_content(prompt)
            analysis_text = response.text.strip()
            if analysis_text.startswith('```json'):
                analysis_text = analysis_text[7:-3]
            elif analysis_text.startswith('```'):
                analysis_text = analysis_text[3:-3]
                
            return json.loads(analysis_text)
        except Exception as e:
            print(f"Error generating overall analysis: {e}")
            return {"summary": "Interview analysis unavailable due to technical issues."}
    
    def start_interview(self):
        print("🤖 AI-Powered Interview Simulator")
        print("=" * 50)
        print("Welcome! I'm your AI interview coach powered by Google Gemini.")
        print("I'll conduct a realistic interview and provide detailed feedback.\n")
        
        self.candidate_name = input("What's your name? ").strip()
        self.position = input("What position are you interviewing for? ").strip()
        
        print(f"\nGreat to meet you, {self.candidate_name}!")
        print(f"Let's practice for your {self.position} interview.\n")
        
        # Choose interview type
        print("What type of interview would you like?")
        print("1. Behavioral interview (STAR method questions)")
        print("2. Technical interview (role-specific technical questions)")
        print("3. General interview (mix of common questions)")
        print("4. Case study interview (problem-solving scenarios)")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        interview_types = {
            "1": "behavioral",
            "2": "technical", 
            "3": "general",
            "4": "case study"
        }
        
        self.interview_type = interview_types.get(choice, "general")
        
        print(f"\n{'='*60}")
        print("🎯 AI INTERVIEW SIMULATION STARTING")
        print(f"{'='*60}")
        print(f"Candidate: {self.candidate_name}")
        print(f"Position: {self.position}")
        print(f"Interview Type: {self.interview_type.title()}")
        print(f"AI Analyst: {'✅ Active' if self.model else '❌ Demo Mode'}")
        print("\nInstructions:")
        print("- Answer each question naturally and thoroughly")
        print("- Type 'next' to move to the next question")
        print("- Type 'feedback' to get detailed analysis of your last answer")
        print("- Type 'end' to finish and get your complete interview analysis")
        print(f"{'='*60}\n")
        
        self.interview_started = True
        input("Press Enter when you're ready to begin your AI interview...")
        
    def conduct_interview(self):
        question_count = 0
        max_questions = 10
        
        while question_count < max_questions:
            # Generate comprehensive context for next question
            context = ""
            if self.responses:
                # Include last 2-3 Q&A pairs for better context
                recent_responses = self.responses[-2:] if len(self.responses) >= 2 else self.responses
                context_parts = []
                for i, resp in enumerate(recent_responses):
                    context_parts.append(f"Q{i+1}: {resp['question']}")
                    context_parts.append(f"A{i+1}: {resp['answer'][:150]}...")
                context = "\n".join(context_parts)
                
                # Add conversation themes for better question variety
                if len(self.responses) >= 3:
                    themes = []
                    for resp in self.responses[-3:]:
                        if 'technical' in resp['question'].lower():
                            themes.append('technical')
                        elif 'team' in resp['question'].lower() or 'collaboration' in resp['question'].lower():
                            themes.append('teamwork')
                        elif 'challenge' in resp['question'].lower() or 'problem' in resp['question'].lower():
                            themes.append('problem-solving')
                    
                    if themes:
                        context += f"\n\nRecent conversation themes: {', '.join(set(themes))}"
            
            # Generate AI question
            print(f"\n🤖 Generating your next question...")
            question = self.generate_question(context)
            
            print(f"\n📋 Question {question_count + 1}:")
            print(f"   {question}")
            print("-" * 60)
            
            start_time = time.time()
            response = input("Your answer: ").strip()
            end_time = time.time()
            
            if response.lower() == 'end':
                break
            elif response.lower() == 'next':
                print("⏭️  Moving to next question...")
                question_count += 1
                continue
            elif response.lower() == 'feedback' and self.responses:
                self.show_detailed_feedback()
                continue
            elif not response:
                print("Please provide an answer or type 'next' to skip.")
                continue
                
            response_time = round(end_time - start_time, 2)
            
            # Analyze response with AI
            print("\n🧠 AI is analyzing your response...")
            analysis = self.analyze_response(question, response)
            
            # Store response with analysis
            response_data = {
                "question": question,
                "answer": response,
                "response_time": response_time,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "analysis": analysis,
                "score": analysis.get('score', 0)
            }
            
            self.responses.append(response_data)
            
            # Show immediate feedback
            print(f"\n✨ AI Feedback:")
            print(f"   Score: {analysis.get('score', 'N/A')}/10")
            print(f"   {analysis.get('feedback', 'Great response!')}")
            
            if analysis.get('follow_up'):
                follow_up = input(f"\n🔄 Follow-up: {analysis['follow_up']}\nYour answer: ").strip()
                if follow_up:
                    response_data['follow_up'] = {
                        'question': analysis['follow_up'],
                        'answer': follow_up
                    }
            
            question_count += 1
            
            if question_count < max_questions:
                continue_choice = input(f"\nReady for question {question_count + 1}? (y/n/end): ").lower()
                if continue_choice == 'n':
                    print("Take your time. Press Enter when ready...")
                    input()
                elif continue_choice == 'end':
                    break
        
        self.end_interview()
    
    def show_detailed_feedback(self):
        if not self.responses:
            print("❌ No responses to analyze yet.")
            return
            
        last_response = self.responses[-1]
        analysis = last_response.get('analysis', {})
        
        print(f"\n📊 DETAILED ANALYSIS - Last Response")
        print("=" * 50)
        print(f"Question: {last_response['question']}")
        print(f"Your Answer: {last_response['answer'][:200]}...")
        print(f"Response Time: {last_response['response_time']}s")
        print(f"Score: {analysis.get('score', 'N/A')}/10")
        print(f"\n💪 Strengths:")
        for strength in analysis.get('strengths', []):
            print(f"   • {strength}")
        print(f"\n🎯 Areas for Improvement:")
        for improvement in analysis.get('improvements', []):
            print(f"   • {improvement}")
        print("=" * 50)
        input("\nPress Enter to continue...")
    
    def end_interview(self):
        print(f"\n{'='*60}")
        print("🏁 AI INTERVIEW ANALYSIS COMPLETE")
        print(f"{'='*60}")
        
        if not self.responses:
            print("No responses to analyze.")
            return
            
        print(f"🧠 Generating comprehensive AI analysis...")
        overall_analysis = self.generate_overall_analysis()
        
        # Show basic stats
        total_score = sum(r.get('score', 0) for r in self.responses)
        avg_score = total_score / len(self.responses) if self.responses else 0
        total_time = sum(r['response_time'] for r in self.responses)
        
        print(f"\n📊 INTERVIEW PERFORMANCE REPORT")
        print("-" * 40)
        print(f"Questions Answered: {len(self.responses)}")
        print(f"Average Score: {avg_score:.1f}/10")
        print(f"Total Time: {total_time:.1f} seconds")
        print(f"Interview Readiness: {overall_analysis.get('interview_readiness', 'Assessment Complete')}")
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   {overall_analysis.get('summary', 'Great job completing the interview!')}")
        
        if overall_analysis.get('top_strengths'):
            print(f"\n💪 TOP STRENGTHS:")
            for strength in overall_analysis['top_strengths']:
                print(f"   ✅ {strength}")
        
        if overall_analysis.get('areas_for_improvement'):
            print(f"\n🎯 AREAS FOR IMPROVEMENT:")
            for area in overall_analysis['areas_for_improvement']:
                print(f"   📈 {area}")
        
        if overall_analysis.get('specific_recommendations'):
            print(f"\n💡 SPECIFIC RECOMMENDATIONS:")
            for rec in overall_analysis['specific_recommendations']:
                print(f"   🔸 {rec}")
        
        if overall_analysis.get('next_steps'):
            print(f"\n🚀 NEXT STEPS:")
            for step in overall_analysis['next_steps']:
                print(f"   📝 {step}")
        
        # Save session
        save_choice = input(f"\nWould you like to save this detailed analysis? (y/n): ")
        if save_choice.lower() == 'y':
            self.save_ai_session(overall_analysis)
            
        print(f"\n🌟 Thank you for using the AI Interview Simulator!")
        print(f"Good luck with your real {self.position} interview! 🍀")
    
    def save_ai_session(self, overall_analysis):
        filename = f"ai_interview_{self.candidate_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        session_data = {
            "candidate_info": {
                "name": self.candidate_name,
                "position": self.position,
                "interview_type": self.interview_type
            },
            "session_metadata": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_questions": len(self.responses),
                "average_score": sum(r.get('score', 0) for r in self.responses) / len(self.responses) if self.responses else 0,
                "ai_powered": self.model is not None
            },
            "responses": self.responses,
            "overall_analysis": overall_analysis
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Complete AI analysis saved as: {filename}")
        except Exception as e:
            print(f"❌ Error saving session: {e}")

def main():
    print("🚀 Starting AI-Powered Interview Simulator...")
    bot = AIInterviewBot()
    
    try:
        bot.start_interview()
        if bot.interview_started:
            bot.conduct_interview()
    except KeyboardInterrupt:
        print("\n\n👋 Interview simulation interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    main()