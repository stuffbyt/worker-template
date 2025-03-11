""" Handler file for model inference. """

import runpod
from model import Model

# Load the model globally so it persists between requests
model = Model()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    
    # Extract parameters with defaults
    prompt = job_input.get('prompt', 'Hello, world!')
    max_length = job_input.get('max_length', None)  # Use env var default if None
    temperature = job_input.get('temperature', None)  # Use env var default if None
    
    try:
        # Generate text using the model
        generated_text = model.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        # Return the results
        return {
            "generated_text": generated_text
        }
    except Exception as e:
        # Return any errors
        return {
            "error": str(e)
        }

# Start the serverless function
runpod.serverless.start({"handler": handler})
