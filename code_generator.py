from openai import OpenAI


def setup_code_generator(api_key, base_url="https://openrouter.ai/api/v1"):
    """Set up the OpenAI client for code generation."""
    return OpenAI(base_url=base_url, api_key=api_key)


def build_context_from_examples(examples):
    """Build a context string from similar examples."""
    context_parts = []
    
    for i, example in enumerate(examples, 1):
        context_parts.append(f"Example {i}:")
        context_parts.append(f"Task: {example['prompt'].strip()}")
        context_parts.append(f"Solution:\n{example['canonical_solution'].strip()}")
        context_parts.append("")  # Empty line between examples
    
    return "\n".join(context_parts)


def create_generation_prompt(task, context):
    """Create the prompt for code generation."""
    return f"""Based on the following examples of Python coding tasks and solutions, generate a complete function for the new task.

{context}

New Task:
{task}

Generate a complete, working Python function that solves this task. Include the function signature and implementation. Only return the code, no explanations."""


def extract_code_from_response(response):
    """Extract code from the LLM response, handling code blocks."""
    if "```python" in response:
        start = response.find("```python") + len("```python")
        end = response.find("```", start)
        return response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        return response[start:end].strip()
    return response.strip()


def generate_code(client, task, examples, model="x-ai/grok-4-fast:free", 
                 max_tokens=500, temperature=0.2):
    """Generate code using the LLM with context from similar examples."""
    print(" Generating code...")
    
    context = build_context_from_examples(examples)
    prompt = create_generation_prompt(task, context)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert Python programmer. Generate clean, efficient, and well-documented code."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    generated_text = response.choices[0].message.content
    return extract_code_from_response(generated_text)