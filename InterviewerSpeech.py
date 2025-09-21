import json
import random
import time
import os
import threading
from datetime import datetime
from typing import List, Dict
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('InterviewerAI.env')

class AIInterviewBot:
    def __init__(self):
        self.candidate_name = ""
        self.position = ""
        self.interview_type = ""
        self.responses = []
        self.interview_started = False
        self.model = None
        self.conversation_history = []
        self.speech_mode = False
        self.use_tts = False
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.tts_engine = None
        
        # Initialize components
        self.setup_gemini_api()
        self.setup_speech_recognition()
        self.setup_text_to_speech()
        
    def setup_gemini_api(self):
        """Setup Gemini API with your API key from .env file"""
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            print("🔑 No API key found in .env file")
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
    
    def setup_speech_recognition(self):
        """Setup speech recognition"""
        try:
            self.microphone = sr.Microphone()
            
            # Test microphone
            print("\n🎤 Setting up speech recognition...")
            with self.microphone as source:
                print("🔧 Adjusting for ambient noise... (please stay quiet)")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("✅ Speech recognition ready!")
            
            # Test recognition
            print("\n🎙️  Speech Test: Say 'hello' to test your microphone...")
            try:
                with self.microphone as source:
                    print("🔴 Listening... (you have 3 seconds)")
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=2)
                test_text = self.recognizer.recognize_google(audio)
                print(f"✅ I heard: '{test_text}'")
                print("🎉 Speech recognition working perfectly!")
            except sr.UnknownValueError:
                print("⚠️  Couldn't understand the test speech, but microphone is working")
            except sr.RequestError as e:
                print(f"⚠️  Speech recognition service error: {e}")
            except Exception as e:
                print(f"⚠️  Speech test failed: {e}")
                
        except Exception as e:
            print(f"❌ Error setting up speech recognition: {e}")
            print("Speech mode will be disabled.")
            self.microphone = None
    
    def setup_text_to_speech(self):
        """Setup text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to use a professional-sounding voice
                for voice in voices:
                    if 'english' in voice.name.lower() and 'female' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            # Set speech rate (words per minute)
            self.tts_engine.setProperty('rate', 180)
            
            # Set volume (0.0 to 1.0)
            self.tts_engine.setProperty('volume', 0.8)
            
            print("✅ Text-to-speech ready!")
            
        except Exception as e:
            print(f"❌ Error setting up text-to-speech: {e}")
            print("Voice output will be disabled.")
            self.tts_engine = None
    
    def speak(self, text):
        """Convert text to speech"""
        if self.use_tts and self.tts_engine:
            try:
                # Clean text for better TTS
                clean_text = text.replace("🤖", "").replace("📋", "").replace("✨", "")
                clean_text = clean_text.replace("-" * 60, "").strip()
                
                if clean_text:
                    self.tts_engine.say(clean_text)
                    self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
    
    def listen_for_speech(self, timeout=30, phrase_limit=None):
        """Listen for speech input and convert to text"""
        if not self.microphone:
            return None
            
        try:
            with self.microphone as source:
                print("🔴 Listening... (speak now)")
                if phrase_limit:
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
                else:
                    audio = self.recognizer.listen(source, timeout=timeout)
            
            print("🧠 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            return text
            
        except sr.WaitTimeoutError:
            print("⏱️  No speech detected within timeout period")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Error with speech recognition service: {e}")
            return None
        except Exception as e:
            print(f"❌ Speech recognition error: {e}")
            return None
    
    def get_user_input(self, prompt="Your answer: ", allow_speech=True):
        """Get input from user via text or speech"""
        if self.speech_mode and allow_speech and self.microphone:
            print(f"\n{prompt}")
            print("💬 Choose your input method:")
            print("1. 🎙️  Speak your answer (press 1)")
            print("2. ⌨️  Type your answer (press 2)")
            print("3. 🔄 Switch to text-only mode (press 3)")
            
            choice = input("\nChoice (1/2/3): ").strip()
            
            if choice == '1':
                print(f"\n🎙️  Ready to record your answer!")
                print("📝 Special commands you can say:")
                print("   - Say 'next question' to skip")
                print("   - Say 'end interview' to finish")
                print("   - Say 'get feedback' to analyze your last answer")
                print("   - Say 'repeat question' to hear the question again")
                
                input("Press Enter when ready to speak...")
                
                speech_text = self.listen_for_speech(timeout=60, phrase_limit=120)
                
                if speech_text:
                    print(f"📝 You said: '{speech_text}'")
                    
                    # Check for voice commands
                    speech_lower = speech_text.lower()
                    if 'next question' in speech_lower or 'skip' in speech_lower:
                        return 'next'
                    elif 'end interview' in speech_lower:
                        return 'end'
                    elif 'get feedback' in speech_lower or 'feedback' in speech_lower:
                        return 'feedback'
                    elif 'repeat question' in speech_lower:
                        return 'repeat'
                    
                    # Confirm the transcription
                    confirm = input("Is this correct? (y/n/retry): ").lower()
                    if confirm == 'y':
                        return speech_text
                    elif confirm == 'retry':
                        return self.get_user_input(prompt, allow_speech)
                    else:
                        print("Let's try typing instead...")
                        return input("Type your answer: ").strip()
                else:
                    print("⚠️  Speech not detected. Let's try typing...")
                    return input("Type your answer: ").strip()
                    
            elif choice == '2':
                return input("Type your answer: ").strip()
            elif choice == '3':
                self.speech_mode = False
                print("✅ Switched to text-only mode")
                return input("Type your answer: ").strip()
            else:
                print("Invalid choice, using text input...")
                return input("Type your answer: ").strip()
        else:
            return input(prompt).strip()

    def generate_question(self, context: str = "") -> str:
        """Generate interview questions using Gemini AI"""
        if not self.model:
            fallback_questions = [
                "Tell me about yourself and your background.",
                "Why are you interested in this position?",
                "What are your greatest strengths?",
                "Describe a challenging project you've worked on.",
                "How do you handle working under pressure?",
                "Tell me about a time you had to work with a difficult team member.",
                "What's your greatest accomplishment in your career so far?",
                "How do you stay updated with industry trends?",
                "Describe a time when you failed and what you learned from it.",
                "Why are you looking to leave your current position?"
            ]
            return random.choice(fallback_questions)
            
        prompt = f"""
        You are an experienced interviewer conducting a {self.interview_type} interview for a {self.position} position.
        Candidate name: {self.candidate_name}
        
        {context}
        
        Generate ONE thoughtful, professional interview question that:
        - Is appropriate for the {self.position} role
        - Follows best interview practices
        - Is different from previously asked questions
        - Encourages detailed responses with specific examples
        - Is suitable for verbal response (clear and conversational)
        - Tests relevant skills for the position
        
        Return ONLY the question, no additional text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating question: {e}")
            return random.choice([
                "Tell me about a time when you overcame a significant challenge.",
                "Describe a project you're particularly proud of.",
                "How do you handle tight deadlines and pressure?"
            ])

    def analyze_response(self, question: str, answer: str, response_time: float) -> Dict:
        """Analyze candidate's response using Gemini AI"""
        if not self.model:
            return {
                "feedback": "API not available. Great answer!",
                "score": random.randint(6, 9),
                "strengths": ["Good communication", "Clear thinking"],
                "improvements": ["Consider more specific examples"],
                "follow_up": None
            }
            
        prompt = f"""
        As an expert interview coach, analyze this candidate's response:
        
        Position: {self.position}
        Interview Type: {self.interview_type}
        Question: {question}
        Answer: {answer}
        Response Time: {response_time} seconds
        
        Note: This answer was given {"verbally" if self.speech_mode else "in text"}.
        
        Provide analysis in this JSON format:
        {{
            "feedback": "Brief encouraging feedback (2-3 sentences)",
            "score": score_out_of_10,
            "strengths": ["strength1", "strength2"],
            "improvements": ["improvement1", "improvement2"],
            "follow_up": "relevant follow-up question or null",
            "key_insights": ["insight1", "insight2"]
        }}
        
        Consider:
        - Content quality and relevance
        - Structure (STAR method for behavioral questions)
        - Specific examples and details
        - Confidence and communication style
        - Response timing (too quick/slow)
        
        Be constructive, specific, and encouraging.
        """
        
        try:
            response = self.model.generate_content(prompt)
            analysis_text = response.text.strip()
            if analysis_text.startswith('```json'):
                analysis_text = analysis_text[7:-3]
            elif analysis_text.startswith('```'):
                analysis_text = analysis_text[3:-3]
                
            analysis = json.loads(analysis_text)
            return analysis
        except Exception as e:
            print(f"Error analyzing response: {e}")
            return {
                "feedback": "Thank you for your detailed response!",
                "score": 7,
                "strengths": ["Clear communication"],
                "improvements": ["Consider adding more specific examples"],
                "follow_up": None,
                "key_insights": ["Shows good thinking"]
            }
    
    def generate_overall_analysis(self) -> Dict:
        """Generate comprehensive interview analysis"""
        if not self.model or not self.responses:
            return {"summary": "Interview completed successfully!"}
            
        # Prepare comprehensive data
        scores = [r.get('score', 0) for r in self.responses]
        avg_score = sum(scores) / len(scores) if scores else 0
        response_times = [r['response_time'] for r in self.responses]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        conversation_summary = "\n".join([
            f"Q{i+1}: {r['question']}\nA: {r['answer'][:150]}...\nScore: {r.get('score', 'N/A')}/10"
            for i, r in enumerate(self.responses[-6:])  # Last 6 for better context
        ])
        
        speech_note = " (conducted with speech input)" if self.speech_mode else ""
        
        prompt = f"""
        Analyze this complete interview performance for a {self.position} position{speech_note}:
        
        Candidate: {self.candidate_name}
        Interview Type: {self.interview_type}
        Questions Answered: {len(self.responses)}
        Average Score: {avg_score:.1f}/10
        Average Response Time: {avg_response_time:.1f} seconds
        
        Recent Q&A History:
        {conversation_summary}
        
        Provide comprehensive analysis in JSON format:
        {{
            "overall_score": score_out_of_10,
            "summary": "Overall performance summary (4-5 sentences)",
            "top_strengths": ["strength1", "strength2", "strength3"],
            "areas_for_improvement": ["area1", "area2", "area3"],
            "specific_recommendations": ["actionable_rec1", "actionable_rec2", "actionable_rec3"],
            "interview_readiness": "Ready/Needs Practice/Needs Significant Work",
            "next_steps": ["concrete_step1", "concrete_step2"],
            "standout_moments": ["moment1", "moment2"],
            "red_flags": ["flag1", "flag2"] or [],
            "industry_specific_advice": "advice specific to the {self.position} role"
        }}
        
        Be honest but encouraging. Focus on actionable feedback that will genuinely help improve interview performance.
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
            return {
                "summary": "Interview analysis unavailable due to technical issues.",
                "overall_score": avg_score,
                "interview_readiness": "Assessment Complete"
            }
    
    def start_interview(self):
        print("🤖 AI-Powered Interview Simulator with Speech Recognition")
        print("=" * 70)
        print("Welcome! I'm your advanced AI interview coach.")
        print("I can help you practice with realistic questions and detailed feedback.\n")
        
        self.candidate_name = input("What's your name? ").strip()
        self.position = input("What position are you interviewing for? ").strip()
        
        print(f"\nGreat to meet you, {self.candidate_name}!")
        print(f"Let's practice for your {self.position} interview.\n")
        
        # Choose interview type
        print("What type of interview would you like?")
        print("1. 🎯 Behavioral interview (STAR method questions)")
        print("2. 💻 Technical interview (role-specific technical questions)")
        print("3. 📋 General interview (mix of common questions)")
        print("4. 🧩 Case study interview (problem-solving scenarios)")
        print("5. 🚀 Leadership interview (management and leadership questions)")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        interview_types = {
            "1": "behavioral",
            "2": "technical", 
            "3": "general",
            "4": "case study",
            "5": "leadership"
        }
        
        self.interview_type = interview_types.get(choice, "general")
        
        # Setup speech preferences
        if self.microphone:
            print(f"\n🎙️  Speech Recognition Setup:")
            speech_choice = input("Enable speech input for answers? (y/n): ").lower()
            self.speech_mode = speech_choice == 'y'
            
            if self.speech_mode and self.tts_engine:
                tts_choice = input("Enable voice output (AI reads questions aloud)? (y/n): ").lower()
                self.use_tts = tts_choice == 'y'
        
        print(f"\n{'='*80}")
        print("🎯 AI INTERVIEW SIMULATION STARTING")
        print(f"{'='*80}")
        print(f"Candidate: {self.candidate_name}")
        print(f"Position: {self.position}")
        print(f"Interview Type: {self.interview_type.title()}")
        print(f"AI Analyst: {'✅ Active (Gemini Pro)' if self.model else '❌ Demo Mode'}")
        print(f"Speech Input: {'✅ Enabled' if self.speech_mode else '❌ Text Only'}")
        print(f"Voice Output: {'✅ Enabled' if self.use_tts else '❌ Text Only'}")
        
        print("\n📋 Instructions:")
        if self.speech_mode:
            print("- 🎙️  Speak your answers naturally (or type if you prefer)")
            print("- 🗣️  Say 'next question' to skip")
            print("- 🗣️  Say 'end interview' to finish")
            print("- 🗣️  Say 'get feedback' for detailed analysis")
        print("- ⌨️  Type commands: 'next', 'feedback', 'end', 'help'")
        print("- 🎯 Answer thoroughly with specific examples")
        print("- ⭐ Use STAR method for behavioral questions (Situation, Task, Action, Result)")
        print(f"{'='*80}\n")
        
        self.interview_started = True
        input("Press Enter when you're ready to begin your AI interview...")
        
    def conduct_interview(self):
        question_count = 0
        max_questions = 12
        current_question = ""
        
        while question_count < max_questions:
            # Generate context for next question
            context = ""
            if self.responses:
                recent_responses = self.responses[-3:]  # Last 3 for context
                context = "Previous questions and responses:\n" + "\n".join([
                    f"Q: {r['question']}\nA: {r['answer'][:100]}...\nScore: {r.get('score', 'N/A')}"
                    for r in recent_responses
                ])
            
            # Generate AI question
            print(f"\n🤖 Generating personalized question {question_count + 1}...")
            current_question = self.generate_question(context)
            
            print(f"\n📋 Question {question_count + 1} of {max_questions}:")
            print(f"   {current_question}")
            print("-" * 80)
            
            # Speak the question if TTS is enabled
            if self.use_tts:
                question_for_speech = f"Question {question_count + 1}: {current_question}"
                self.speak(question_for_speech)
            
            start_time = time.time()
            response = self.get_user_input()
            end_time = time.time()
            
            # Handle special commands
            if response.lower() == 'end':
                break
            elif response.lower() == 'next':
                print("⏭️  Skipping to next question...")
                question_count += 1
                continue
            elif response.lower() == 'repeat':
                if self.use_tts:
                    self.speak(current_question)
                continue
            elif response.lower() == 'feedback' and self.responses:
                self.show_detailed_feedback()
                continue
            elif response.lower() == 'help':
                self.show_help()
                continue
            elif not response or len(response.strip()) < 5:
                print("⚠️  Please provide a more detailed answer (at least 5 characters).")
                continue
                
            response_time = round(end_time - start_time, 2)
            
            # Analyze response with AI
            print("\n🧠 AI is analyzing your response...")
            analysis = self.analyze_response(current_question, response, response_time)
            
            # Store response with analysis
            response_data = {
                "question": current_question,
                "answer": response,
                "response_time": response_time,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "analysis": analysis,
                "score": analysis.get('score', 0),
                "input_method": "speech" if self.speech_mode else "text"
            }
            
            self.responses.append(response_data)
            
            # Show immediate feedback
            score = analysis.get('score', 'N/A')
            feedback = analysis.get('feedback', 'Great response!')
            
            print(f"\n✨ AI Feedback:")
            print(f"   📊 Score: {score}/10")
            print(f"   💬 {feedback}")
            
            # Show key insights if available
            if analysis.get('key_insights'):
                insights = analysis['key_insights'][:2]  # Show top 2 insights
                print(f"   🔍 Key Insights: {' • '.join(insights)}")
            
            # Speak feedback if TTS enabled
            if self.use_tts:
                speech_feedback = f"Score: {score} out of 10. {feedback}"
                self.speak(speech_feedback)
            
            # Handle follow-up questions
            if analysis.get('follow_up'):
                print(f"\n🔄 Follow-up: {analysis['follow_up']}")
                if self.use_tts:
                    self.speak(f"Follow-up question: {analysis['follow_up']}")
                    
                follow_up = self.get_user_input("Follow-up answer: ")
                if follow_up and follow_up.lower() not in ['next', 'end', 'feedback']:
                    response_data['follow_up'] = {
                        'question': analysis['follow_up'],
                        'answer': follow_up
                    }
            
            question_count += 1
            
            # Progress indicator
            progress = (question_count / max_questions) * 100
            print(f"\n📈 Progress: {question_count}/{max_questions} ({progress:.0f}%)")
            
            if question_count < max_questions:
                continue_choice = input(f"\nReady for question {question_count + 1}? (y/n/end): ").lower()
                if continue_choice == 'n':
                    print("Take your time. Press Enter when ready...")
                    input()
                elif continue_choice == 'end':
                    break
        
        self.end_interview()
    
    def show_help(self):
        """Show help information"""
        print(f"\n{'='*60}")
        print("📚 HELP - Available Commands")
        print(f"{'='*60}")
        print("During the interview, you can use these commands:")
        print("  • 'next' or 'skip' - Move to next question")
        print("  • 'end' - Finish interview and get analysis")
        print("  • 'feedback' - Get detailed analysis of your last answer")
        print("  • 'repeat' - Hear the current question again (if TTS enabled)")
        print("  • 'help' - Show this help message")
        
        if self.speech_mode:
            print("\nSpeech Commands (say these out loud):")
            print("  • 'next question' - Skip to next")
            print("  • 'end interview' - Finish interview")  
            print("  • 'get feedback' - Analyze last answer")
            print("  • 'repeat question' - Hear question again")
        
        print("\n💡 Tips for Better Interviews:")
        print("  • Use specific examples and stories")
        print("  • Follow STAR method (Situation, Task, Action, Result)")
        print("  • Speak clearly and at moderate pace")
        print("  • Take a moment to think before answering")
        print("  • Show enthusiasm and confidence")
        print(f"{'='*60}")
        input("\nPress Enter to continue...")
    
    def show_detailed_feedback(self):
        """Show detailed feedback for the last response"""
        if not self.responses:
            print("❌ No responses to analyze yet.")
            return
            
        last_response = self.responses[-1]
        analysis = last_response.get('analysis', {})
        
        print(f"\n{'='*70}")
        print("📊 DETAILED ANALYSIS - Last Response")
        print(f"{'='*70}")
        print(f"❓ Question: {last_response['question']}")
        print(f"💬 Your Answer: {last_response['answer'][:300]}{'...' if len(last_response['answer']) > 300 else ''}")
        print(f"⏱️  Response Time: {last_response['response_time']}s")
        print(f"📱 Input Method: {last_response.get('input_method', 'text').title()}")
        print(f"📊 Score: {analysis.get('score', 'N/A')}/10")
        
        print(f"\n💪 Strengths:")
        for strength in analysis.get('strengths', ['Good response']):
            print(f"   ✅ {strength}")
            
        print(f"\n🎯 Areas for Improvement:")
        for improvement in analysis.get('improvements', ['Keep practicing']):
            print(f"   📈 {improvement}")
            
        if analysis.get('key_insights'):
            print(f"\n🔍 Key Insights:")
            for insight in analysis['key_insights']:
                print(f"   💡 {insight}")
        
        print(f"{'='*70}")
        
        if self.use_tts:
            speech_feedback = f"Your score was {analysis.get('score', 'unknown')} out of 10. {analysis.get('feedback', '')}"
            self.speak(speech_feedback)
            
        input("\nPress Enter to continue...")
    
    def end_interview(self):
        """End the interview and provide comprehensive analysis"""
        print(f"\n{'='*80}")
        print("🏁 AI INTERVIEW ANALYSIS COMPLETE")
        print(f"{'='*80}")
        
        if not self.responses:
            print("No responses to analyze.")
            return
            
        print(f"🧠 Generating comprehensive AI analysis...")
        overall_analysis = self.generate_overall_analysis()
        
        # Calculate statistics
        scores = [r.get('score', 0) for r in self.responses]
        total_score = sum(scores)
        avg_score = total_score / len(scores) if scores else 0
        total_time = sum(r['response_time'] for r in self.responses)
        avg_time = total_time / len(self.responses) if self.responses else 0
        speech_responses = sum(1 for r in self.responses if r.get('input_method') == 'speech')
        
        # Performance report
        print(f"\n📊 INTERVIEW PERFORMANCE REPORT")
        print("-" * 50)
        print(f"📋 Questions Answered: {len(self.responses)}")
        print(f"🎙️  Speech Responses: {speech_responses}/{len(self.responses)}")
        print(f"⭐ Average Score: {avg_score:.1f}/10")
        print(f"⏱️  Total Time: {total_time:.1f} seconds")
        print(f"📈 Average Response Time: {avg_time:.1f} seconds")
        print(f"🎯 Interview Readiness: {overall_analysis.get('interview_readiness', 'Assessment Complete')}")
        
        # Overall assessment
        summary = overall_analysis.get('summary', 'Great job completing the interview!')
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   {summary}")
        
        # Speak summary if TTS enabled
        if self.use_tts:
            summary_speech = f"Interview complete! Your average score was {avg_score:.1f} out of 10. {summary}"
            self.speak(summary_speech)
        
        # Top strengths
        if overall_analysis.get('top_strengths'):
            print(f"\n💪 TOP STRENGTHS:")
            for i, strength in enumerate(overall_analysis['top_strengths'], 1):
                print(f"   {i}. ✅ {strength}")
        
        # Areas for improvement
        if overall_analysis.get('areas_for_improvement'):
            print(f"\n🎯 AREAS FOR IMPROVEMENT:")
            for i, area in enumerate(overall_analysis['areas_for_improvement'], 1):
                print(f"   {i}. 📈 {area}")
        
        # Specific recommendations
        if overall_analysis.get('specific_recommendations'):
            print(f"\n💡 SPECIFIC RECOMMENDATIONS:")
            for i, rec in enumerate(overall_analysis['specific_recommendations'], 1):
                print(f"   {i}. 🔸 {rec}")
        
        # Standout moments
        if overall_analysis.get('standout_moments'):
            print(f"\n🌟 STANDOUT MOMENTS:")
            for moment in overall_analysis['standout_moments']:
                print(f"   ⭐ {moment}")
        
        # Red flags (if any)
        if overall_analysis.get('red_flags') and overall_analysis['red_flags']:
            print(f"\n⚠️  AREAS OF CONCERN:")
            for flag in overall_analysis['red_flags']:
                print(f"   🚩 {flag}")
        
        # Industry-specific advice
        if overall_analysis.get('industry_specific_advice'):
            print(f"\n🏢 INDUSTRY-SPECIFIC ADVICE:")
            print(f"   💼 {overall_analysis['industry_specific_advice']}")
        
        # Next steps
        if overall_analysis.get('next_steps'):
            print(f"\n🚀 NEXT STEPS:")
            for i, step in enumerate(overall_analysis['next_steps'], 1):
                print(f"   {i}. 📝 {step}")
        
        # Save session
        save_choice = input(f"\nWould you like to save this detailed analysis? (y/n): ")
        if save_choice.lower() == 'y':
            self.save_ai_session(overall_analysis)
            
        print(f"\n🌟 Thank you for using the AI Interview Simulator!")
        print(f"Good luck with your real {self.position} interview! 🍀")
    
    def save_ai_session(self, overall_analysis):
        """Save the interview session to a JSON file"""
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
                "ai_powered": self.model is not None,
                "speech_mode": self.speech_mode,
                "tts_enabled": self.use_tts
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
    print("🚀 Starting AI-Powered Interview Simulator with Speech Recognition...")
    bot = AIInterviewBot()
    
    try:
        bot.start_interview()
        if bot.interview_started:
            bot.conduct_interview()
    except KeyboardInterrupt:
        print("\n\n👋 Interview simulation interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()