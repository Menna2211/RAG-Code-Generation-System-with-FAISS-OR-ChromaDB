from datasets import load_dataset

def load_humaneval_dataset():
    """Load the HumanEval dataset with coding problems."""
    print("Loading HumanEval dataset...")
    
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    examples = []
    for item in dataset:
        examples.append({
            'task_id': item['task_id'],
            'prompt': item['prompt'],
            'canonical_solution': item['canonical_solution'],
            'entry_point': item['entry_point']
        })
    
    print(f"✓ Loaded {len(examples)} coding examples")
    return examples


def get_prompts_from_examples(examples):
    """Extract just the prompt text from examples."""
    return [ex['prompt'] for ex in examples]


def find_example_by_id(examples, example_id):
    """Find a specific example by task ID."""
    for example in examples:
        if example['task_id'] == example_id:
            return example
    return None