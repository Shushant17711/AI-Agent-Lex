"""
File system tools for AMAS.
This module provides tools for interacting with the file system.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileOperationTools:
    """A class providing file operation tools for the multi-agent system."""
    
    def __init__(self, base_dir: Optional[str] = None, file_focus_callback: Optional[callable] = None, file_content_callback: Optional[callable] = None):
        """Initialize with optional base directory and callbacks.

        Args:
            base_dir: Base directory for file operations. If None, uses current working directory.
            file_focus_callback: Optional callback when an agent focuses on a file (path: str).
            file_content_callback: Optional callback when file content is updated (path: str, content: str).
        """
        self.base_dir = base_dir or os.getcwd()
        self.file_focus_callback = file_focus_callback
        self.file_content_callback = file_content_callback
        logger.info(f"File operations will use base directory: {self.base_dir}")
        logger.debug(f"FileOperationTools initialized with focus_callback: {bool(self.file_focus_callback)}, content_callback: {bool(self.file_content_callback)}") # DEBUG LOG
    
    def file_read(self, filename: str) -> Dict[str, Any]:
        """Read content from a file.
        
        Args:
            filename: Path to the file to read.
            
        Returns:
            Dict containing status and either content or error message.
        """
        try:
            full_path = self._get_full_path(filename)
            logger.info(f"Reading file: {full_path}")
            if self.file_focus_callback:
                logger.debug(f"Calling file_focus_callback for READ: {full_path}") # DEBUG LOG
                try:
                    self.file_focus_callback(full_path)
                except Exception as cb_err:
                    logger.warning(f"Error in file_focus_callback during read: {cb_err}")
            else:
                logger.debug("file_focus_callback is None during READ.") # DEBUG LOG

            if not os.path.exists(full_path):
                logger.warning(f"File not found: {full_path}")
                return {"status": "error", "message": f"File {filename} does not exist"}
            
            with open(full_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            logger.info(f"Successfully read file: {full_path} ({len(content)} bytes)")
            return {"status": "success", "content": content}
        except Exception as e:
            logger.error(f"Error reading file {filename}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def file_write(self, filename: str, content: str, mode: str = "w") -> Dict[str, Any]:
        """Write content to a file.
        
        Args:
            filename: Path to the file to write.
            content: Content to write to the file.
            mode: File open mode ('w' for write, 'a' for append).
            
        Returns:
            Dict containing status and message.
        """
        try:
            full_path = self._get_full_path(filename)
            logger.info(f"Writing to file: {full_path} with mode: {mode}")
            if self.file_focus_callback:
                logger.debug(f"Calling file_focus_callback for WRITE: {full_path}") # DEBUG LOG
                try:
                    self.file_focus_callback(full_path)
                except Exception as cb_err:
                    logger.warning(f"Error in file_focus_callback during write: {cb_err}")
            else:
                 logger.debug("file_focus_callback is None during WRITE.") # DEBUG LOG

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, mode, encoding='utf-8') as file:
                file.write(content)
            
            # Verify the file was created
            if os.path.exists(full_path):
                file_size = os.path.getsize(full_path)
                logger.info(f"Successfully wrote to file: {full_path} ({file_size} bytes)")
                # Call content callback after successful write
                if self.file_content_callback:
                    logger.debug(f"Calling file_content_callback after WRITE: {full_path}") # DEBUG LOG
                    try:
                        # For 'w', the content passed is the new full content
                        # For 'a', we need to read the file to get the full content
                        final_content = content
                        if mode == 'a':
                             logger.debug("Reading file content for append callback...") # DEBUG LOG
                             with open(full_path, 'r', encoding='utf-8') as f_read:
                                 final_content = f_read.read()

                        self.file_content_callback(full_path, final_content)
                    except Exception as cb_err:
                        logger.warning(f"Error in file_content_callback after write: {cb_err}")
                else:
                    logger.debug("file_content_callback is None after WRITE.") # DEBUG LOG

                return {
                    "status": "success",
                    "message": f"File {filename} written successfully ({file_size} bytes)"
                }
            else:
                logger.error(f"Failed to verify file exists after writing: {full_path}")
                return {"status": "error", "message": f"Failed to verify file {filename} exists"}
        except Exception as e:
            logger.error(f"Error writing to file {filename}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def file_append(self, filename: str, content: str) -> Dict[str, Any]:
        """Append content to a file.
        
        Args:
            filename: Path to the file to append to.
            content: Content to append to the file.
            
        Returns:
            Dict containing status and message.
        """
        return self.file_write(filename, content, mode="a")
    
    def file_delete(self, filename: str) -> Dict[str, Any]:
        """Delete a file.
        
        Args:
            filename: Path to the file to delete.
            
        Returns:
            Dict containing status and message.
        """
        try:
            full_path = self._get_full_path(filename)
            logger.info(f"Deleting file: {full_path}")
            if self.file_focus_callback:
                logger.debug(f"Calling file_focus_callback for DELETE: {full_path}") # DEBUG LOG
                try:
                    self.file_focus_callback(full_path)
                except Exception as cb_err:
                    logger.warning(f"Error in file_focus_callback during delete: {cb_err}")
            else:
                logger.debug("file_focus_callback is None during DELETE.") # DEBUG LOG

            if not os.path.exists(full_path):
                logger.warning(f"File not found for deletion: {full_path}")
                return {"status": "error", "message": f"File {filename} does not exist"}
            
            os.remove(full_path)
            
            # Verify the file was deleted
            if not os.path.exists(full_path):
                logger.info(f"Successfully deleted file: {full_path}")
                # Call content callback with empty content after successful delete
                if self.file_content_callback:
                    logger.debug(f"Calling file_content_callback after DELETE: {full_path}") # DEBUG LOG
                    try:
                        self.file_content_callback(full_path, "")
                    except Exception as cb_err:
                        logger.warning(f"Error in file_content_callback after delete: {cb_err}")
                else:
                    logger.debug("file_content_callback is None after DELETE.") # DEBUG LOG
                return {"status": "success", "message": f"File {filename} deleted successfully"}
            else:
                logger.error(f"Failed to delete file: {full_path}")
                return {"status": "error", "message": f"Failed to delete file {filename}"}
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def file_exists(self, filename: str) -> Dict[str, Any]:
        """Check if a file exists.
        
        Args:
            filename: Path to the file to check.
            
        Returns:
            Dict containing status and result.
        """
        try:
            full_path = self._get_full_path(filename)
            exists = os.path.exists(full_path)
            # Call focus callback even when just checking existence
            if self.file_focus_callback:
                logger.debug(f"Calling file_focus_callback for EXISTS: {full_path}") # DEBUG LOG
                try:
                    self.file_focus_callback(full_path)
                except Exception as cb_err:
                    logger.warning(f"Error in file_focus_callback during exists check: {cb_err}")
            else:
                logger.debug("file_focus_callback is None during EXISTS check.") # DEBUG LOG

            if exists:
                file_size = os.path.getsize(full_path)
                logger.info(f"File exists: {full_path} ({file_size} bytes)")
                return {
                    "status": "success", 
                    "exists": True,
                    "size": file_size,
                    "message": f"File {filename} exists ({file_size} bytes)"
                }
            else:
                logger.info(f"File does not exist: {full_path}")
                return {
                    "status": "success", 
                    "exists": False,
                    "message": f"File {filename} does not exist"
                }
        except Exception as e:
            logger.error(f"Error checking if file {filename} exists: {str(e)}")
            return {"status": "error", "message": str(e)}
    

    def file_list(self, path: str = '.', recursive: bool = False) -> Dict[str, Any]:
        """List files and directories within a specified path relative to the base directory.

        Args:
            path: Relative path to the directory to list.
            recursive: If True, list files and directories recursively.

        Returns:
            Dict containing status and a list of files/directories (relative to the input path) or error message.
        """
        try:
            full_path = self._get_full_path(path)
            logger.info(f"Listing files in directory: {full_path}")

            if not os.path.exists(full_path):
                logger.warning(f"Directory not found: {full_path}")
                return {"status": "error", "message": f"Directory {path} does not exist"}

            if not os.path.isdir(full_path):
                logger.warning(f"Path is not a directory: {full_path}")
                return {"status": "error", "message": f"Path {path} is not a directory"}

            results = []
            if recursive:
                for root, dirs, files in os.walk(full_path):
                    # Calculate relative path from the starting 'full_path'
                    relative_root = os.path.relpath(root, full_path)
                    if relative_root == '.':
                        relative_root = '' # Avoid './' prefix for top-level items

                    for name in dirs:
                        results.append(os.path.join(relative_root, name))
                    for name in files:
                        results.append(os.path.join(relative_root, name))
            else:
                results = os.listdir(full_path)

            logger.info(f"Successfully listed {len(results)} items in: {full_path} (Recursive: {recursive})")
            # Ensure consistent path separators (use forward slashes)
            results = [item.replace('\\', '/') for item in results]
            return {"status": "success", "files": results}
        except Exception as e:
            logger.error(f"Error listing files in {path}: {str(e)}")
            return {"status": "error", "message": str(e)}


    def apply_diff(self, filename: str, diff: str) -> Dict[str, Any]:
        """Apply changes to a file using a specific diff format.

        Args:
            filename: Path to the file to modify.
            diff: String containing the diff blocks in the specified format:
                  <<<<<<< SEARCH
                  :start_line:START
                  :end_line:END
                  -------
                  [content to find]
                  =======
                  [content to replace with]
                  >>>>>>> REPLACE

        Returns:
            Dict containing status and message.
        """
        full_path = self._get_full_path(filename)
        logger.info(f"Applying diff to file: {full_path}")

        if not os.path.exists(full_path):
            logger.warning(f"File not found for applying diff: {full_path}")
            return {"status": "error", "message": f"File {filename} does not exist"}

        try:
            # Read original content
            with open(full_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            import re

            def visualize_whitespace(text):
                """Replace spaces with '·' and tabs with '→' for clearer diffs."""
                return text.replace(' ', '·').replace('\t', '→')

            # Parse all diff blocks first
            diff_blocks_raw = re.split(r'<<<<<<< SEARCH', diff)[1:]
            parsed_blocks = []
            for i, block_raw in enumerate(diff_blocks_raw):
                block_index = i + 1
                if '=======' not in block_raw or '>>>>>>> REPLACE' not in block_raw:
                    raise ValueError(f"Block {block_index}: Invalid format: Missing '=======' or '>>>>>>> REPLACE'")

                try:
                    header_search, rest = block_raw.split('-------', 1)
                    search_content_raw, replace_content_part = rest.split('=======', 1)
                    replace_content_raw, _ = replace_content_part.split('>>>>>>> REPLACE', 1)
                except ValueError:
                    raise ValueError(f"Block {block_index}: Invalid format: Structure mismatch around '-------', '=======', or '>>>>>>> REPLACE'")

                start_line_match = re.search(r':start_line:(\d+)', header_search)
                end_line_match = re.search(r':end_line:(\d+)', header_search)

                if not start_line_match or not end_line_match:
                    raise ValueError(f"Block {block_index}: Invalid format: Missing start_line or end_line")

                start_line = int(start_line_match.group(1))
                end_line = int(end_line_match.group(1))
                if start_line <= 0 or end_line < 0 or start_line > end_line + 1: # end_line can be 0 if replacing line 1
                     raise ValueError(f"Block {block_index}: Invalid line numbers: start={start_line}, end={end_line}")

                # Store raw content initially, strip later during comparison/application
                parsed_blocks.append({
                    'index': block_index,
                    'start_line': start_line,
                    'end_line': end_line,
                    'search_content_raw': search_content_raw,
                    'replace_content_raw': replace_content_raw
                })

            # --- Single Pass: Validate and Apply ---
            offset = 0 # Tracks line number changes
            logger.debug(f"Starting single-pass apply_diff for {len(parsed_blocks)} blocks.")

            for block in parsed_blocks:
                block_index = block['index']
                original_start_line = block['start_line']
                original_end_line = block['end_line']

                # Calculate current indices based on offset from previous changes
                current_start_index = original_start_line - 1 + offset
                # End index for slicing is exclusive, so use original_end_line directly
                current_end_index = original_end_line + offset

                logger.debug(f"Block {block_index}: Original lines {original_start_line}-{original_end_line}. Applying at offset {offset} -> indices [{current_start_index}:{current_end_index}]")

                # Bounds check against the current state of 'lines'
                if current_start_index < 0 or current_end_index > len(lines) or current_start_index > current_end_index:
                    raise IndexError(f"Block {block_index}: Calculated indices [{current_start_index}:{current_end_index}] are out of bounds for current file length {len(lines)} (original lines {original_start_line}-{original_end_line}, offset {offset}).")

                # Extract the actual slice from the file *at this moment*
                actual_slice_lines = lines[current_start_index:current_end_index]
                actual_content_joined = "".join(actual_slice_lines)

                # Prepare search content from the diff block (normalize and strip)
                search_content_target_normalized = block['search_content_raw'].replace('\r\n', '\n').strip()

                # --- More Robust Comparison: Normalize and strip entire blocks ---
                actual_content_normalized = actual_content_joined.replace('\r\n', '\n').strip()

                # Direct string comparison after normalization
                if search_content_target_normalized != actual_content_normalized:
                    # --- ADD DETAILED LOGGING FOR MISMATCH ---
                    logger.error("--- APPLY DIFF MISMATCH DETAILS ---")
                    logger.error(f"Block {block_index} @ Original Lines: {original_start_line}-{original_end_line}")
                    logger.error(f"Effective Indices (0-based): [{current_start_index}:{current_end_index}]")
                    # Log RAW content
                    logger.error(f"--- DIFF SEARCH BLOCK (RAW from LLM): ---")
                    logger.error(repr(block['search_content_raw']))
                    logger.error(f"--- ACTUAL FILE SLICE (RAW): ---")
                    logger.error(repr(actual_content_joined))
                    # Log the normalized versions being compared
                    logger.error(f"--- DIFF SEARCH BLOCK (Normalized & Stripped): ---")
                    logger.error(repr(search_content_target_normalized))
                    logger.error(f"--- ACTUAL FILE SLICE (Normalized & Stripped): ---")
                    logger.error(repr(actual_content_normalized))
                    logger.error("--- END APPLY DIFF MISMATCH DETAILS ---")
                    # --- END ADDED LOGGING ---

                    # Simplified error message
                    error_msg = (
                        f"Block {block_index}: Content mismatch applying diff to {filename} at original lines {original_start_line}-{original_end_line} "
                        f"(effective range {current_start_index + 1}-{current_end_index}). Normalized content does not match. See logs for details."
                    )
                    logger.error(f"Apply diff failed: {error_msg}")
                    return {"status": "error", "message": error_msg}
                # --- End Robust Comparison ---

                # --- Content matches (after normalization), proceed with replacement ---

                # Prepare replacement content (strip outer whitespace/newlines)
                replace_content_stripped = block['replace_content_raw'].strip('\r\n')
                replace_lines = replace_content_stripped.splitlines(keepends=True) # Keep internal newlines

                # Handle edge case: replacing with empty content
                if not replace_lines and replace_content_stripped == "":
                    replace_lines = []
                # Ensure trailing newline consistency (important!)
                elif replace_lines and not replace_lines[-1].endswith(('\n', '\r')):
                    # Add newline if:
                    # 1. The original block being replaced ended with a newline OR
                    # 2. The replacement content itself has multiple lines.
                    original_block_had_trailing_newline = actual_content_joined.endswith(('\n', '\r'))
                    replacement_has_multiple_lines = len(replace_lines) > 1

                    if original_block_had_trailing_newline or replacement_has_multiple_lines:
                        # Use the presumed newline style of the file (check last line of original slice)
                        newline_char = '\n' # Default
                        if actual_slice_lines:
                            last_line = actual_slice_lines[-1]
                            if last_line.endswith('\r\n'):
                                newline_char = '\r\n'
                            elif last_line.endswith('\r'):
                                newline_char = '\r' # Less common but possible
                        logger.debug(f"Block {block_index}: Adding trailing newline '{repr(newline_char)}' to replacement.")
                        replace_lines[-1] += newline_char

                logger.debug(f"Block {block_index}: Replacing lines {current_start_index + 1}-{current_end_index} ({len(actual_slice_lines)} lines) with {len(replace_lines)} lines.")

                # Apply the replacement to the 'lines' list
                lines[current_start_index:current_end_index] = replace_lines

                # Update the offset for the *next* block
                lines_removed = len(actual_slice_lines)
                lines_added = len(replace_lines)
                offset += (lines_added - lines_removed)
                logger.debug(f"Block {block_index} applied. Lines added: {lines_added}, removed: {lines_removed}. New offset: {offset}")

            logger.info(f"All {len(parsed_blocks)} diff blocks applied successfully.")

            # Write modified content back
            modified_content = "".join(lines)
            with open(full_path, 'w', encoding='utf-8') as file:
                file.write(modified_content)

            file_size = os.path.getsize(full_path)
            logger.info(f"Successfully applied diff to file: {full_path} ({file_size} bytes)")

            # Call content callback after successful diff apply
            if self.file_content_callback:
                logger.debug(f"Calling file_content_callback after APPLY_DIFF: {full_path}")
                try:
                    self.file_content_callback(full_path, modified_content)
                except Exception as cb_err:
                    logger.warning(f"Error in file_content_callback after apply_diff: {cb_err}")
            else:
                logger.debug("file_content_callback is None after APPLY_DIFF.")

            return {
                "status": "success",
                "message": f"Diff applied successfully to {filename} ({file_size} bytes)"
            }

        except ValueError as ve:
             logger.error(f"Invalid diff format for file {filename}: {str(ve)}")
             return {"status": "error", "message": f"Invalid diff format: {str(ve)}"}
        except IndexError as ie:
             logger.error(f"Line number error applying diff to file {filename}: {str(ie)}")
             return {"status": "error", "message": f"Line number error applying diff: {str(ie)}"}
        except Exception as e:
            import traceback
            logger.error(f"Error applying diff to file {filename}: {str(e)}\n{traceback.format_exc()}")
            return {"status": "error", "message": f"Error applying diff: {str(e)}"}

    def _get_full_path(self, filename: str) -> str:
        """Get the full, validated path for a file relative to the base directory.

        Args:
            filename: Relative path to a file. Absolute paths are disallowed.

        Returns:
            Full absolute path to the file within the base directory.

        Raises:
            ValueError: If the path is absolute or attempts to traverse outside the base directory.
        """
        if os.path.isabs(filename):
             # Disallow absolute paths provided by the agent for security
             raise ValueError(f"Absolute paths are not allowed: {filename}")

        # Normalize base_dir and the combined path
        # os.path.abspath ensures consistent format and resolves '.' or '..' if base_dir itself contains them
        resolved_base_dir = os.path.abspath(self.base_dir)
        
        # Join and then resolve the combined path
        combined_path = os.path.join(resolved_base_dir, filename)
        resolved_combined_path = os.path.abspath(combined_path)

        # Check if the resolved path is within the base directory
        # os.path.commonpath is a reliable way to check containment after resolving
        if os.path.commonpath([resolved_base_dir, resolved_combined_path]) != resolved_base_dir:
            raise ValueError(f"Path traversal attempt detected: '{filename}' resolves outside the base directory '{resolved_base_dir}'")

        return resolved_combined_path
