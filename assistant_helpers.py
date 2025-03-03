# Assistant API Helpers
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import io
import json
import pandas as pd

# Get API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

# Configure OPENAI client
client = OpenAI(api_key = OPENAI_API_KEY)

# Function for converting CSV to dict for querying
def process_csv(file_path):
    
    # Load CSV into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Convert to a dictionary
    data_dict = df.to_dict(orient="records")  # List of dicts (row-based)

    # Output as JSON string
    return json.dumps(data_dict)

# Create assistant
def create_assistant(name='City Helper', instructions='', model='gpt-4o', temperature=1, enable_function=False):
    if enable_function:
        tool_array = [{"type": "code_interpreter"},
                      {
                        "type": "function",
                        "function": {
                            "name": "process_csv",
                            "description": "Converts an uploaded CSV file into a dictionary for structured querying.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_id": {"type": "string", "description": "The ID of the uploaded CSV file."}
                                },
                                "required": ["file_id"],
                            },
                        },
                    },
                ]
    else:
        tool_array = [{"type": "code_interpreter"}]
    
    assistant = client.beta.assistants.create(
        instructions=instructions,
        name="City Helper",
        tools= tool_array,
        model="gpt-4o",
        temperature=temperature
    )
    return assistant

# Prompt an assistant
def prompt_assistant(assistant=None, prompt=None, file_path=None, debug=False):
    # Check for presence of assistant
    if assistant is None or prompt is None:
        print("Error: Assistant and/or prompt missing")
        return
    
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

        if debug:
            print(f"Run Status: {run_status.status}")

        if run_status.status == "requires_action":
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            for tool_call in tool_calls:
                if tool_call.function.name == "process_csv":
                    # file_id = tool_call.function.arguments["file_id"]
                    
                    # Process the CSV
                    print(f"Processing CSV file: {file.id}")
                    csv_dict = process_csv(file_path)

                    # Submit the function output
                    client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=[{"tool_call_id": tool_call.id, "output": csv_dict}],
                    )
                    print(f"Function output submitted for too call id: {tool_call.id}")
        elif run_status.status in ["completed", "failed", "cancelled"]:
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
            timestamp = msg.created_at 
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
                # Check if this is a function call or a code interpreter call
                if hasattr(tool_call, "code_interpreter"):
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

                elif hasattr(tool_call, "function"):  # Handle function calls correctly
                    tool_entry = {
                        "type": "function_call",
                        "timestamp": timestamp,
                        "content": f"\nFunction Call: {tool_call.function.name} with arguments {tool_call.function.arguments}",
                    }
                    entries.append(tool_entry)

    # Sort all messages and tool calls by timestamp
    sorted_entries = sorted(entries, key=lambda x: x["timestamp"])

    # Construct ordered output
    output = "\n".join(entry["content"] for entry in sorted_entries)

    return output, thread, messages, run_steps

# Evaluate method
def evaluate_method(assistant=None, prompt=None, file_path=None, debug=False, runs=10):
    
    # Create array to store responses
    responses = []

    # Loop and acquire outputs
    for i in range(runs):
        response = prompt_assistant(assistant=assistant, prompt=prompt, file_path=file_path)
        responses.append(response[2][0].content[0].text.value)
        print(f'Run {i+1}/{runs} complete.')

    # Check answers against LLM judge
    accuracy = []
    for response in responses:
        completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "You are tasked with determining whether or not an answer is correct and complete.\
            The user will provide an input and you will compare it to your sample answer and output 1 if it contains all \
            of the information in the sample answer and 0 otherwise.  Your sample answer is below:\
            New York, because of its vibrant city life and diversity. because of its vibrant city life and diversity. Additionally, It's home to the largest metropolitan zoo in the US.\
            You will ONLY output 0 or 1 and nothing else."},
            {"role": "user", "content": response}
        ]
        )
        accuracy.append([response, completion.choices[0].message.content])
        print(f'Evaluation {i+1}/{runs} complete.')
    
    accuracy_df = pd.DataFrame(accuracy, columns=['Response','Accurate'])

    return accuracy, accuracy_df
    