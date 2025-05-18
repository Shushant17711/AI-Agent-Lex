"""
Asynchronous Communication Bus for AMAS agents using asyncio.

Provides a mechanism for agents to send and receive messages asynchronously
within the same process without blocking.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple, TypedDict, Literal, Union # Added TypedDict, Literal, Union
import uuid # For generating unique request IDs
import time # For timeout calculations

logger = logging.getLogger(__name__)

# --- Common Message Types (Recommendation 9) ---
# Define common message types as constants for consistency and clarity
MT_REQUEST = "request"
MT_RESPONSE = "response"
MT_UPDATE = "update"
MT_DATA = "data"
MT_ERROR = "error"
MT_SYSTEM = "system" # For internal bus/orchestrator messages
# Add other specific types as needed, e.g., MT_REQUEST_HELP = "request_help"

# Define a standard message format using TypedDict for better structure and type checking
class Message(TypedDict, total=False):
    """
    Represents the structure of a message exchanged via the communication bus.
    Use constants like MT_REQUEST, MT_RESPONSE for the 'type' field where applicable.
    """
    sender: str          # Name of the sending agent (must be registered)
    recipient: str       # Name of the receiving agent or 'broadcast'
    type: str            # Type of message (e.g., MT_REQUEST, MT_RESPONSE, MT_UPDATE)
    payload: Dict[str, Any] # The actual content/data of the message
    request_id: Optional[str] # Unique ID for request messages (present if type=MT_REQUEST)
    response_to_id: Optional[str] # ID of the request this message is responding to

# Example usage hint (not part of the code):
# msg: Message = {"sender": "agent1", "recipient": "agent2", "type": "query", "payload": {"data": 123}}

class CommunicationBus:
    """
    An asynchronous in-process message queue for inter-agent communication using asyncio.

    Note: This bus is designed for use within a single asyncio event loop. Accessing
    the bus concurrently from multiple threads without external locking mechanisms
    may lead to race conditions or unexpected behavior, despite the use of asyncio primitives.
    """

    def __init__(self, max_queue_size: int = 0):
        """
        Initializes the communication bus.

        Args:
            max_queue_size: The maximum number of messages an agent's queue can hold.
                            0 means infinite size (default).
        """
        self._queues: Dict[str, asyncio.Queue] = {}
        self._agent_names: set[str] = set()
        self._max_queue_size = max_queue_size
        logger.info(f"Async Communication Bus initialized (max_queue_size={max_queue_size}).")

    def register_agent(self, agent_name: str):
        """Registers an agent, creating a dedicated asyncio message queue for it."""
        if agent_name not in self._queues:
            self._queues[agent_name] = asyncio.Queue(maxsize=self._max_queue_size)
            self._agent_names.add(agent_name)
            logger.info(f"Agent '{agent_name}' registered with the async communication bus.")
        else:
            logger.warning(f"Agent '{agent_name}' attempted to register again.")

    async def send_message(self, sender: Optional[str] = None, recipient: Optional[str] = None, message_type: Optional[str] = None, payload: Optional[Dict[str, Any]] = None, *, message: Optional[Message] = None):
        """Asynchronously sends a message to a specific agent or broadcasts it.

        Can either construct the message from sender, recipient, type, and payload,
        OR send a pre-constructed message object.

        Args:
            sender: The name of the sending agent (required if not providing `message`).
            recipient: The name of the recipient agent or 'broadcast' (required if not providing `message`).
            message_type: The type of message (e.g., MT_REQUEST) (required if not providing `message`).
            payload: The data associated with the message.
            message: A pre-constructed Message object to send directly. If provided,
                     sender, recipient, message_type, and payload args are ignored.

        Returns:
            For broadcast messages: A list of agent names for whom sending failed.
            For direct messages: True if sending was successful, False otherwise.
            None if validation fails before sending.
        """
        constructed_message: Optional[Message] = None # Use a different name to avoid confusion
        if message is None:
            # --- Construct message from arguments ---
            if not all([sender, recipient, message_type]):
                 logger.error("send_message requires sender, recipient, and message_type if message object is not provided.")
                 return
            if sender not in self._agent_names:
                logger.error(f"Validation failed: Unregistered agent '{sender}' attempted to send a message.")
                return

            constructed_message = Message(
                sender=sender,
                recipient=recipient,
                type=message_type,
                payload=payload or {}
            )
            # Validate the constructed message (basic check)
            if not self._validate_message(constructed_message):
                 # Validation logs the error
                 return None # Indicate validation failure
        else:
            # --- Use the provided message object ---
            constructed_message = message # Use the message passed in
            # Validate the provided message
            if not self._validate_message(constructed_message):
                 # Validation logs the error
                 return None # Indicate validation failure

        # --- Proceed with sending the validated message ---
        recipient = constructed_message["recipient"] # Already validated to exist
        sender = constructed_message["sender"] # Already validated to exist
        message_to_send = constructed_message # Use the validated message

        # --- Broadcast Logic ---
        if recipient == "broadcast":
            failed_recipients: List[str] = []
            logger.debug(f"Broadcasting message from '{sender}' (Type: {message_to_send['type']})")
            tasks = []
            for agent_name, q in self._queues.items():
                if agent_name != sender: # Don't send broadcast to self
                    try:
                        # Use put_nowait for broadcast to avoid blocking if a queue is full (less critical for broadcast)
                        # Or create tasks for await q.put(message) if guaranteed delivery is needed
                        # Use put_nowait for broadcast: don't block sender if one queue is full.
                        q.put_nowait(message_to_send)
                    except asyncio.QueueFull:
                         logger.warning(f"Broadcast failed for '{agent_name}': Queue full (max_size={self._max_queue_size}). Message from '{sender}' dropped.")
                         failed_recipients.append(agent_name)
                    except Exception as e:
                        logger.error(f"Broadcast failed for '{agent_name}': Exception putting message in queue: {e}")
                        failed_recipients.append(agent_name)
            return failed_recipients # Return list of agents for whom broadcast failed

        # --- Direct Message Logic ---
        elif recipient in self._queues:
            logger.debug(f"Sending message from '{sender}' to '{recipient}' (Type: {message_to_send['type']})")
            try:
                # Use await for direct messages to wait if the queue is full (respecting max_queue_size)
                await self._queues[recipient].put(message_to_send)
                return True # Indicate success
            except asyncio.QueueFull: # Should not happen if max_queue_size=0, but handle defensively
                 logger.error(f"Failed to send message to '{recipient}': Queue is full (max_size={self._max_queue_size}).")
                 return False # Indicate failure
            except Exception as e:
                 logger.error(f"Failed to put message in queue for '{recipient}': {e}")
                 return False # Indicate failure
        else:
            logger.warning(f"Cannot send message: Recipient agent '{recipient}' not found or not registered.")
            return False # Indicate failure

    def _validate_message(self, message: Message) -> bool:
        """Basic validation for a message object."""
        required_keys = ["sender", "recipient", "type"]
        for key in required_keys:
            if key not in message or not message[key]: # Check presence and non-empty
                logger.error(f"Message validation failed: Missing or empty required key '{key}'. Message: {message}")
                return False

        sender = message["sender"]
        if sender not in self._agent_names:
             logger.error(f"Message validation failed: Sender '{sender}' is not registered. Message: {message}")
             return False

        # Add more specific validation if needed (e.g., check recipient validity, type format)
        return True

    async def get_message(self, agent_name: str, timeout: Optional[float] = None) -> Optional[Message]:
        """Asynchronously retrieves the next message for a specific agent.

        Args:
            agent_name: The name of the agent retrieving the message.
            timeout: Optional maximum time in seconds to wait. If None, waits indefinitely.
                     If 0, returns immediately (non-blocking).

        Returns:
            The next message dictionary, or None if the queue is empty (and timeout=0)
            or the timeout is reached.
        """
        if agent_name not in self._queues:
            logger.error(f"Agent '{agent_name}' attempted to get messages but is not registered.")
            return None

        try:
            if timeout == 0: # Non-blocking check
                message = self._queues[agent_name].get_nowait()
            elif timeout is None: # Wait forever
                 message = await self._queues[agent_name].get()
            else: # Wait with timeout
                 message = await asyncio.wait_for(self._queues[agent_name].get(), timeout=timeout)

            if message:
                 # Mark task as done after retrieving
                 self._queues[agent_name].task_done()
                 logger.debug(f"Agent '{agent_name}' retrieved message: {message}")
                 return message
            else: # Should not happen with await get() unless queue is closed, but handle defensively
                 return None

        except asyncio.QueueEmpty: # Only happens with get_nowait()
            # logger.debug(f"No messages available for agent '{agent_name}'.") # Can be noisy
            return None
        except asyncio.TimeoutError:
            logger.debug(f"Timeout waiting for message for agent '{agent_name}'.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving message for agent '{agent_name}': {e}")
            return None

    def get_agent_names(self) -> List[str]:
        """Returns a list of registered agent names."""
        return list(self._agent_names)

    async def join(self, agent_name: Optional[str] = None):
        """Waits until all items in the specified agent's queue (or all queues) have been received and processed.

        Args:
            agent_name: If specified, waits only for this agent's queue. Otherwise, waits for all queues.
        """
        if agent_name:
            if agent_name in self._queues:
                await self._queues[agent_name].join()
                logger.info(f"Queue for agent '{agent_name}' joined.")
            else:
                logger.warning(f"Cannot join queue for unknown agent '{agent_name}'.")
        else:
            tasks = [q.join() for q in self._queues.values()]
            await asyncio.gather(*tasks)
            logger.info("All agent queues joined.")

    def unregister_agent(self, agent_name: str):
        """
        Unregisters an agent, removing its message queue.

        Args:
            agent_name: The name of the agent to unregister.
        """
        if agent_name in self._queues:
            # Potentially wait for the queue to be empty before removing?
            # Or just remove it? Current implementation removes immediately.
            # Consider implications if messages are still being processed.
            del self._queues[agent_name]
            self._agent_names.discard(agent_name)
            logger.info(f"Agent '{agent_name}' unregistered and queue removed.")
        else:
            logger.warning(f"Attempted to unregister non-existent agent '{agent_name}'.")

    async def send_request_and_wait(
        self,
        sender: str,
        recipient: str,
        request_type: str,
        request_payload: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0  # Default timeout of 30 seconds
    ) -> Optional[Message]:
        """
        Sends a request message to a specific agent and waits for a corresponding response.

        Args:
            sender: The name of the sending agent.
            recipient: The name of the recipient agent.
            request_type: The type of request being sent.
            request_payload: The data associated with the request.
            timeout: Maximum time in seconds to wait for the response.

        Returns:
            The response message dictionary, or None if no response is received within the timeout
            or if an error occurs.
        """
        if sender not in self._agent_names:
            logger.error(f"Unregistered agent '{sender}' attempted to send a request.")
            return None
        if recipient not in self._queues:
            logger.warning(f"Recipient agent '{recipient}' for request not found or not registered.")
            return None
        if recipient == "broadcast":
             logger.error(f"Cannot use send_request_and_wait with 'broadcast' recipient.")
             return None

        request_id = str(uuid.uuid4())
        request_message: Message = {
            "sender": sender,
            "recipient": recipient,
            "type": request_type,
            "payload": request_payload or {},
            "request_id": request_id
        }

        try:
            # Send the complete request message using the modified send_message
            await self.send_message(message=request_message)
            logger.debug(f"Agent '{sender}' sent request (ID: {request_id}) to '{recipient}'. Waiting for response...")
        except Exception as e:
            logger.error(f"Failed to send request (ID: {request_id}) from '{sender}' to '{recipient}': {e}")
            return None

        unrelated_messages: List[Message] = []
        target_response: Optional[Message] = None

        async def _wait_loop():
            nonlocal target_response # Allow modification of outer scope variable
            while True: # Loop indefinitely until response found or timeout occurs externally
                # Wait indefinitely for *any* message for the sender
                message = await self.get_message(sender, timeout=None)
                if message:
                    if message.get("response_to_id") == request_id:
                        logger.info(f"Agent '{sender}' received target response (ID: {request_id}) from '{message.get('sender')}'.")
                        target_response = message
                        return # Exit the wait loop successfully
                    else:
                        # Buffer unrelated messages
                        logger.debug(f"Agent '{sender}' received unrelated message while waiting for {request_id}. Buffering: {message}")
                        unrelated_messages.append(message)
                # If get_message returns None unexpectedly (e.g., queue closed), loop might break or error
                # asyncio.sleep(0.01) # Small sleep to prevent tight loop if get_message behaves unexpectedly

        try:
            # Use asyncio.wait_for for clean timeout handling
            await asyncio.wait_for(_wait_loop(), timeout=timeout)
            # If wait_for completes without timeout, target_response should be set
            return target_response
        except asyncio.TimeoutError:
            logger.warning(f"Agent '{sender}' timed out waiting for response (ID: {request_id}) from '{recipient}' after {timeout}s.")
            return None
        except Exception as e:
            logger.error(f"Error while agent '{sender}' was waiting for response (ID: {request_id}): {e}")
            return None
        finally:
            # After waiting (success or timeout), put buffered messages back into the queue
            if unrelated_messages:
                logger.debug(f"Agent '{sender}' putting back {len(unrelated_messages)} buffered message(s) after waiting for {request_id}.")
                sender_queue = self._queues.get(sender)
                if sender_queue:
                    # Put messages back at the front? No, asyncio.Queue doesn't support that easily.
                    # Put them at the back using put_nowait.
                    for msg in reversed(unrelated_messages): # Put back in reverse order received? Or original order? Let's try original.
                         try:
                             sender_queue.put_nowait(msg)
                         except asyncio.QueueFull:
                             logger.error(f"Agent '{sender}' queue is full. Failed to put back buffered message: {msg}. This message is lost!")
                         except Exception as e_put:
                             logger.error(f"Error putting back buffered message for agent '{sender}': {e_put}. Message lost: {msg}")
                else:
                    logger.error(f"Agent '{sender}' queue not found. Cannot put back {len(unrelated_messages)} buffered messages.")