"""Gemini LLM Service implementation for AMAS."""

import os
import logging
import queue
import threading
import random
import time
import google.generativeai as genai
from typing import Dict, Any, Optional

# Assuming a BaseLLMService interface exists like this:
# from .llm_service import BaseLLMService
from abc import ABC, abstractmethod

class BaseLLMService(ABC):
    """Abstract base class for LLM services."""
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generates a response from the LLM."""
        pass

logger = logging.getLogger(__name__)

class GeminiLLMService(BaseLLMService):
    """LLM Service implementation using Google Gemini."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash",
                 max_retries: int = 1, retry_delay: int = 3, default_timeout: float = 60.0, **kwargs):
        """
        Initializes the Gemini LLM Service.

        Args:
            api_key: Google AI API Key. If None, attempts to read from
                     GOOGLE_API_KEY environment variable.
            model_name: The specific Gemini model to use (e.g., "gemini-1.5-flash", "gemini-1.5-pro-latest").
            max_retries: Maximum number of retries on failure (default: 1).
            retry_delay: Base delay in seconds between retries (default: 3).
            default_timeout: Default timeout in seconds for API calls (default: 60.0).
            **kwargs: Additional configuration options for the generative model
                      (e.g., temperature, top_p). These become the base generation_config.
        """
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise ValueError("Gemini API key not provided and GOOGLE_API_KEY environment variable not set.")
            
        genai.configure(api_key=resolved_api_key)
        
        self.model_name = model_name
        # Store the initial kwargs as the base generation config
        self.base_generation_config_dict = kwargs
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.default_timeout = default_timeout
        
        try:
            # Pass the initial config dict directly during model initialization if supported,
            # otherwise, we'll merge later. Let's assume we merge later for flexibility.
            self.model = genai.GenerativeModel(self.model_name) 
            logger.info(f"GeminiLLMService initialized with model: {self.model_name}")
            logger.info(f"Base generation config: {self.base_generation_config_dict}")
            # Optional: Test connection 
            # self.model.generate_content("test", generation_config=genai.types.GenerationConfig(**self.base_generation_config_dict, max_output_tokens=5))
            # logger.info("Gemini API connection successful.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model '{self.model_name}': {e}")
            raise ConnectionError(f"Failed to initialize Gemini model '{self.model_name}': {e}") from e

    def _threaded_generate(self, result_queue: queue.Queue, prompt: str, config: Optional[genai.types.GenerationConfig]):
        """Internal function to run the API call in a thread."""
        try:
            response = self.model.generate_content(prompt, generation_config=config)
            result_queue.put(response)
        except Exception as e:
            result_queue.put(e) # Put the exception in the queue

    def generate_response(self, prompt: str, timeout: Optional[float] = None, **kwargs) -> str:
        """
        Generates a response from the configured Gemini model with timeout and retries.

        Args:
            prompt: The input prompt for the model.
            timeout: Timeout in seconds for this specific API call. Overrides default_timeout.
            **kwargs: Overrides for generation config for this specific call
                      (e.g., temperature).

        Returns:
            The generated text response.

        Raises:
            TimeoutError: If the API call exceeds the specified timeout.
            RuntimeError: If the API call fails after all retries.
            ValueError: If the response structure is invalid or blocked.
        """
        call_timeout = timeout if timeout is not None else self.default_timeout
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Sending prompt to Gemini model '{self.model_name}' (Attempt {attempt + 1}/{self.max_retries + 1}, Timeout: {call_timeout}s):\n{prompt[:200]}...")

                # Merge base config with call-specific overrides
                merged_config_dict = {**self.base_generation_config_dict, **kwargs}
                logger.debug(f"Effective generation config for this call: {merged_config_dict}")

                current_generation_config = None
                if merged_config_dict: # Only create config object if there are params
                    try:
                        current_generation_config = genai.types.GenerationConfig(**merged_config_dict)
                    except TypeError as config_e:
                        logger.error(f"Failed to create GenerationConfig from merged dict {merged_config_dict}: {config_e}. Using model defaults.")
                        current_generation_config = None

                # --- Make the API call with timeout ---
                result_queue = queue.Queue()
                api_thread = threading.Thread(
                    target=self._threaded_generate,
                    args=(result_queue, prompt, current_generation_config),
                    daemon=True # Allows program to exit even if thread is running
                )
                api_thread.start()

                try:
                    # Wait for the result with timeout
                    result = result_queue.get(timeout=call_timeout)
                    if isinstance(result, Exception):
                        raise result # Re-raise exception caught in the thread
                    response = result # Successful result from the queue
                except queue.Empty:
                    # Timeout occurred
                    logger.warning(f"Gemini API call timed out after {call_timeout} seconds on attempt {attempt + 1}.")
                    # We don't necessarily need to kill the thread, but log it.
                    # The library might eventually finish or error out internally.
                    raise TimeoutError(f"Gemini API call timed out after {call_timeout} seconds.")
                except Exception as thread_e: # Catch exceptions raised from the queue
                    logger.warning(f"Exception received from API thread on attempt {attempt + 1}: {thread_e}")
                    raise thread_e # Re-raise to be handled by outer try-except
                finally:
                    # Ensure the queue is cleared in case of unexpected state
                    while not result_queue.empty():
                        try: result_queue.get_nowait()
                        except queue.Empty: break
                # ------------------------------------

                # Handle potential safety blocks or empty responses
                if not response.candidates:
                    logger.warning("Gemini response has no candidates.")
                    finish_reason = getattr(response, 'prompt_feedback', {}).get('block_reason')
                    if finish_reason:
                        # Don't retry on content blocked, return error immediately
                        return f"[ERROR: Content blocked due to {finish_reason}]"
                    if hasattr(response, 'parts') and response.parts:
                        logger.warning(f"Response has parts but no candidates: {response.parts}")
                    # Treat as potentially retryable error if no specific block reason
                    # Raise a specific error type if possible, otherwise generic Exception
                    raise ValueError("No candidates received in Gemini response and no block reason provided.")


                # Accessing text, handling potential errors if response structure is unexpected
                generated_text = ""
                try:
                    # Accessing the text content safely from the first candidate
                    if response.candidates[0].content and response.candidates[0].content.parts:
                        generated_text = response.candidates[0].content.parts[0].text
                    else:
                        # Log finish reason if available
                        finish_reason = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN')
                        # Treat unexpected structure/empty content as potentially retryable
                        raise ValueError(f"Unexpected Gemini response structure or empty content in candidate 0. Finish reason: {finish_reason}")

                except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
                    # Treat parsing errors or the ValueError from above as potentially retryable
                    logger.warning(f"Error processing Gemini response on attempt {attempt + 1}: {e}. Raw Response: {getattr(response, '__dict__', str(response))}")
                    # Raise the error to trigger the retry logic below
                    raise ValueError(f"Error processing Gemini response: {e}") from e # Wrap for clarity


                logger.debug(f"Received successful response from Gemini on attempt {attempt + 1}: {generated_text[:200]}...")
                return generated_text # Success, exit the loop and return

            except genai.types.BlockedPromptException as e:
                # Handle content policy violations
                logger.warning(f"Content blocked: {e}")
                # Don't retry on content blocked, return error immediately
                return f"[ERROR: Content blocked due to {e}]"
            except (TimeoutError, ConnectionError, genai.types.InternalServerError, genai.types.ServiceUnavailableError) as e:
                # Handle timeouts, network issues, and specific retryable Google API errors
                last_exception = e
                logger.warning(f"Retryable error encountered on attempt {attempt + 1}: {type(e).__name__} - {e}")
                # Retry logic will handle below
            except Exception as e:
                # Handle other unexpected errors
                last_exception = e
                logger.warning(f"Gemini API call failed on attempt {attempt + 1}/{self.max_retries + 1}: {type(e).__name__} - {str(e)}")
                if attempt < self.max_retries:
                   # Exponential backoff with jitter
                   delay = min(self.retry_delay * (2 ** attempt), 60) # Cap at 60 seconds
                   jitter = random.uniform(0.5, 1.0) * delay # Add jitter
                   logger.info(f"Waiting {jitter:.2f} seconds before retrying...")
                   time.sleep(jitter)
                else:
                   logger.error(f"Gemini API call failed after {self.max_retries + 1} attempts.")
                   # Propagate the last encountered error for the agent to handle
                   raise RuntimeError(f"Gemini API call failed after {self.max_retries + 1} attempts. Last error: {type(last_exception).__name__} - {str(last_exception)}") from last_exception

        logger.error("Exited retry loop unexpectedly without returning or raising.")
        raise RuntimeError(f"Gemini API call failed unexpectedly after retries. Last error: {type(last_exception).__name__} - {str(last_exception)}") from last_exception