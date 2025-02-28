# Assistant API Helpers
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# Get API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

# Configure OPENAI client
client = OpenAI(api_key = OPENAI_API_KEY)

# Create assistant
def create_assistant(name='City Helper', instructions='', model='gpt-4o'):
    assistant = client.beta.assistants.create(
        instructions="",
        name="City Helper",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4o"
    )
    return assistant

# # Prompt an assistant
# def prompt_assistant(assistant=None, prompt=None, file=None):
#     # Check for presence of assistant
#     if assistant==None or prompt==None:
#         print('Error: Assistant and/or prompt missing')
#         return 
    
#     # Upload file
#     file = client.files.create(
#         file=open("tse_takehome_dataset.csv", "rb"),
#         purpose="assistants"
#     )
    
#     # Create Thread
#     thread = client.beta.threads.create()

#     # Add message
#     message = client.beta.threads.messages.create(
#         thread_id=thread.id,
#         role="user",
#         content=prompt,
#         attachments= [{
#             "file_id": file.id,
#             "tools": [{"type": "code_interpreter"}] 
#         }]
#     )

#     # Start run to process thread
#     run = client.beta.threads.runs.create(
#         thread_id = thread.id,
#         assistant_id = assistant.id
#     )

#     # Poll until the run is completed
#     while True:
#         run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        
#         if run_status.status in ["completed", "failed", "cancelled"]:
#             break  # Stop when the run is finished
        
#         time.sleep(2)  # Wait a bit before checking again

#     output = ''

#     # Fetch and print the latest assistant response
#     messages = client.beta.threads.messages.list(thread_id=thread.id)
#     for msg in reversed(messages.data):  # Reverse to get the latest response first
#         if msg.role == "assistant":
#             output += "Assistant Response: " + msg.content[0].text.value + '\n'
    
#     # Fetch and print steps
#     run_steps = client.beta.threads.runs.steps.list(
#         thread_id=thread.id, 
#         run_id=run.id
#     )
#     for step in run_steps.data:
#         if step.step_details.type == "tool_calls":
#             for tool_call in step.step_details.tool_calls:
#                 # if tool_call.type == "code_interpreter":
#                 output += "\nTool Call Input:\n" + tool_call.code_interpreter.input + '\n'
#                 if tool_call.code_interpreter.outputs:
#                         logs_data = tool_call.code_interpreter.outputs[0].logs
#                         output += "\nTool Call Output:\n" + tool_call.code_interpreter.outputs[0].logs + '\n'

#     return output, thread

# Prompt an assistant
def prompt_assistant(assistant=None, prompt=None, file_path=None):
    # Check for presence of assistant
    if assistant is None or prompt is None:
        print("Error: Assistant and/or prompt missing")
        return

    # Upload file
    file = client.files.create(
        file=open(file_path, "rb"),
        purpose="assistants",
    )

    # Create Thread
    thread = client.beta.threads.create()

    # Add message
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
        attachments=[{"file_id": file.id, "tools": [{"type": "code_interpreter"}]}],
    )

    # Start run to process thread
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    # Poll until the run is completed
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        if run_status.status in ["completed", "failed", "cancelled"]:
            break  # Stop when the run is finished

        time.sleep(2)  # Wait a bit before checking again

    # Fetch messages and tool call steps
    messages = client.beta.threads.messages.list(thread_id=thread.id).data
    run_steps = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id).data

    # Extract timestamps to properly interleave outputs
    entries = []

    # Process assistant messages
    for msg in messages:
        if msg.role == "assistant":
            timestamp = msg.created_at  # Assuming the API provides a timestamp
            entries.append(
                {
                    "type": "assistant",
                    "timestamp": timestamp,
                    "content": "\nAssistant Response: " + msg.content[0].text.value,
                }
            )

    # Process tool call outputs
    for step in run_steps:
        if step.step_details.type == "tool_calls":
            timestamp = step.created_at  # Assuming tool calls also have timestamps
            for tool_call in step.step_details.tool_calls:
                tool_entry = {
                    "type": "tool_call",
                    "timestamp": timestamp,
                    "content": "\nTool Call Input:\n" + tool_call.code_interpreter.input,
                }
                entries.append(tool_entry)

                if tool_call.code_interpreter.outputs:
                    logs = tool_call.code_interpreter.outputs[0].logs
                    output_entry = {
                        "type": "tool_output",
                        "timestamp": timestamp,
                        "content": "\nTool Call Output:\n" + logs + '\n',
                    }
                    entries.append(output_entry)

    # Sort all messages and tool calls by timestamp
    sorted_entries = sorted(entries, key=lambda x: x["timestamp"])

    # Construct ordered output
    output = "\n".join(entry["content"] for entry in sorted_entries)

    return output, thread, messages, run_steps
