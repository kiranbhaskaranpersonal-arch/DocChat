import asyncio
from google.adk.agents import Agent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

# 1. SETUP (Run this cell once)
ollama_model = LiteLlm(model="ollama_chat/llama3.2")

creator = Agent(
    name="creator",
    model=ollama_model,
    instruction="Write a LinkedIn post. If feedback exists, improve it.",
    output_key="current_draft"
)

critic = Agent(
    name="critic",
    model=ollama_model,
    instruction="Review 'current_draft'. If it's perfect, say 'TERMINATE'. Otherwise, give 1 tip.",
    output_key="critic_feedback"
)

refinement_loop = LoopAgent(
    name="linkedin_team",
    sub_agents=[creator, critic],
    max_iterations=2
)

# Initialize the runner globally
runner = InMemoryRunner(agent=refinement_loop)

# ---------------------------------------------------------
# 2. THE FOOL-PROOF EXECUTION ENGINE
# ---------------------------------------------------------
class AgentManager:
    def __init__(self, runner):
        self.runner = runner
        self.active_session_id = None

    async def get_valid_session(self):
        """Always returns a valid session ID, even if the runner reset."""
        # If we don't have an ID, or if the runner throws an error, make a new one
        if self.active_session_id is None:
            new_session = await self.runner.session_service.create_session(
                app_name="linkedin_gen", 
                user_id="kiran_user"
            )
            self.active_session_id = new_session.id
        return self.active_session_id

    async def run(self, topic: str):
        # Ensure we have a valid ID before starting
        sid = await self.get_valid_session()
        
        user_input = types.Content(
            role='user', 
            parts=[types.Part.from_text(text=f"Topic: {topic}")]
        )

        try:
            print(f"📡 Using Session: {sid}")
            async for event in self.runner.run_async(
                user_id="kiran_user", 
                session_id=sid, 
                new_message=user_input
            ):
                if event.is_final_response():
                    print(f"\n✨ DONE:\n{event.content.parts[0].text}")
        
        except Exception as e:
            # If the session is NOT found, we clear our local ID and try ONE more time
            if "SessionNotFoundError" in str(e):
                print("⚠️ Session expired/not found. Auto-generating a new one and retrying...")
                self.active_session_id = None # Reset
                await self.run(topic) # Recursive retry
            else:
                raise e

# 3. RUN IT
# Every time you run this cell, it will work—even if you restarted the runner!
manager = AgentManager(runner)
await manager.run("The future of AI agents in 2026")